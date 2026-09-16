"""One-step document batching: modal run batch_benchmark.py"""
import json
import time
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path

import modal
from modal_app import image, cache, secret, MODEL, REVISION, hf_auth
GPU_RATE = 0.001097  # Modal H100 USD/second, checked 2026-09-16

app = modal.App('open-jev-batching')
image = image.add_local_python_source('modal_app')
ROOT = Path(__file__).parent


@app.function(image=image, secrets=[secret], volumes={'/cache': cache}, gpu='H100', timeout=900)
def sweep(cases):
    started = time.perf_counter()
    import gc
    import itertools
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, LogitsProcessorList
    from json_canvas import FinalReadoutCanvas, constrained_sampler

    token = hf_auth()
    revision = REVISION
    processor = AutoProcessor.from_pretrained(MODEL, revision=revision, token=token)
    processor.tokenizer.padding_side = 'left'
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, revision=revision, token=token, dtype=torch.bfloat16, device_map='cuda').eval()
    cache.commit()
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    groups, layouts, lengths = {}, {}, {}
    for case in cases:
        name = case['experiment']
        groups.setdefault(name, []).append(case)
        instructions = ('For each question below, return a boolean answer in a JSON object. '
                        'Use exactly the listed keys, in this order. No explanations.\n'
                        + '\n'.join(json.dumps(k) + ': ' + q for k, q in case['questions'].items()))
        case['prompt'] = '<user_text>\n' + case['document'] + '\n</user_text>\n\n<instructions>\n' + instructions + '\n</instructions>'
    for name, rows in groups.items():
        keys = list(rows[0]['questions'])
        texts = [json.dumps(dict(zip(keys, vs))) for vs in itertools.product((False, True), repeat=len(keys))]
        layout = FinalReadoutCanvas(processor.tokenizer, texts, model.config.canvas_length,
                                   model.config.text_config.vocab_size, eos, model.device)
        # The same initial noise per document as the seed-0 serial baseline.
        # A single shared noise row prevents batch size from changing RNG assignments.
        torch.manual_seed(0)
        initial = layout.initialize_canvas(1, model.device)
        layout.initialize_canvas = lambda batch_size, device, initial=initial: initial.expand(batch_size, -1).clone()
        layouts[name] = layout
        lengths[name] = max(processor.apply_chat_template(
            [{'role': 'user', 'content': c['prompt']}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')['input_ids'].shape[1] for c in rows)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    resident_gib = torch.cuda.memory_allocated() / 2**30
    records, predictions, failures, warmups = [], [], [], []

    def run(rows):
        start = time.perf_counter()
        name = rows[0]['experiment']; layout = layouts[name]; layout.steps = 0
        inputs = processor.apply_chat_template(
            [[{'role': 'user', 'content': c['prompt']}] for c in rows],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
            padding='max_length', max_length=lengths[name], truncation=False).to(model.device)
        assert inputs['input_ids'].shape == (len(rows), lengths[name])
        torch.cuda.reset_peak_memory_stats()
        with constrained_sampler(model, layout):
            torch.cuda.synchronize(); gen_start = time.perf_counter()
            model.generate(**inputs, max_new_tokens=model.config.canvas_length, max_denoising_steps=1,
                           cache_implementation='dynamic', logits_processor=LogitsProcessorList([layout]))
            torch.cuda.synchronize(); generation_seconds = time.perf_counter() - gen_start
        values = [layout.decode(ids)[1] for ids in layout.select_final().tolist()]
        probabilities = {}
        for key, pos, scores in zip(rows[0]['questions'], layout.variable_positions, layout.last_value_logits):
            ix = next(i for i, t in enumerate(layout.columns[pos]) if processor.tokenizer.decode([t]).strip() == 'true')
            probabilities[key] = scores.softmax(-1)[:, ix].cpu().tolist()
        seconds = time.perf_counter() - start
        outputs = [{'id': c['id'], 'experiment': name, 'values': value,
                    'uncalibrated_p_true': {k: p[i] for k, p in probabilities.items()},
                    'reference_p_true': c['reference'], 'labels': c['labels']}
                   for i, (c, value) in enumerate(zip(rows, values))]
        return {'experiment': name, 'documents': len(rows), 'request_seconds': seconds,
                'generation_seconds': generation_seconds, 'input_padded_length': lengths[name],
                'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                'device_free_gib_after': torch.cuda.mem_get_info()[0]/2**30}, outputs

    for size in (64, 32, 16, 8, 4, 2, 1):
        pools = [[rows[i % len(rows)] for i in range(64)] for rows in groups.values()]
        chunks = [rows[i:i+size] for rows in pools for i in range(0, len(rows), size)]
        try:
            # Warm every distinct (schema, batch size, padded prompt length) shape.
            seen = set()
            for chunk in chunks:
                shape = (chunk[0]['experiment'], len(chunk))
                if shape not in seen:
                    stat, _ = run(chunk); stat['requested_batch_size'] = size
                    warmups.append(stat); seen.add(shape)
            for repeat in range(3):
                for chunk in chunks:
                    stat, out = run(chunk)
                    stat.update(requested_batch_size=size, repeat=repeat)
                    records.append(stat)
                    predictions.extend(dict(x, requested_batch_size=size, repeat=repeat) for x in out)
            print(f'Batch {size}: completed three passes over 128 repeated document slots; stopping at first fit', flush=True)
            break
        except torch.cuda.OutOfMemoryError as error:
            failures.append({'requested_batch_size': size, 'error': str(error),
                             'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
            for layout in layouts.values(): layout.last_value_logits = None
            gc.collect(); torch.cuda.empty_cache()
            print(f'Batch {size}: out of GPU memory; halving', flush=True)
            continue
    return {'model': MODEL, 'revision': revision, 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'hardware': torch.cuda.get_device_name(),
            'device_total_gib': torch.cuda.get_device_properties(0).total_memory/2**30,
            'resident_allocated_gib_after_setup': resident_gib, 'setup_seconds': setup_seconds,
            'remote_function_seconds': time.perf_counter()-started,
            'batch_records': records, 'predictions': predictions, 'failures': failures, 'warmups': warmups,
            'padding': 'left; fixed maximum input length per experiment at every batch size',
            'noise': 'identical seed-0 vocabulary noise row for each document at every batch size',
            'cost_scope': 'GPU-only rate estimate; excludes CPU/RAM/storage and lifecycle outside function',
            'workload': '64 repeated slots per experiment (128 total); 32 unique documents. Stop at first size that fits, starting at 64.'}


def summarize(result):
    summaries=[]
    baseline={(x['experiment'], x['id']): x['values'] for x in result['predictions']
              if x['requested_batch_size']==1 and x['repeat']==0}
    if not baseline:
        baseline = {(x['experiment'], x['id']): x['values'] for x in result.get('historical_unbatched_records', [])}
    for size in sorted({r['requested_batch_size'] for r in result['batch_records']}):
        batches=[r for r in result['batch_records'] if r['requested_batch_size']==size]
        rows=[r for r in result['predictions'] if r['requested_batch_size']==size]
        if len(rows)!=384: continue  # Exclude partially completed/OOM settings.
        total=sum(r['request_seconds'] for r in batches)
        first=list({(r['experiment'],r['id']):r for r in rows if r['repeat']==0}.values())
        retrieval=[r for r in first if r['experiment']=='code-rag']
        support=[r for r in first if r['experiment']=='customer-voice']
        ranks=[]
        for key in retrieval[0]['values']:
            ranking=sorted(retrieval,key=lambda r:(-r['uncalibrated_p_true'][key],r['id']))
            ranks.append(next(i for i,r in enumerate(ranking,1) if r['labels'][key]))
        summaries.append({'requested_batch_size':size, 'actual_batch_sizes':sorted({r['documents'] for r in batches}),
            'documents_measured':len(rows), 'documents_per_second':len(rows)/total,
            'inconsistent_repeated_documents': sum(len({json.dumps(r['values'], sort_keys=True) for r in rows if (r['experiment'], r['id'])==key}) > 1 for key in {(r['experiment'], r['id']) for r in rows}),
            'amortized_ms_per_document':1000*total/len(rows),
            'median_batch_latency_ms':statistics.median(r['request_seconds'] for r in batches)*1000,
            'gpu_usd_per_1000_documents':1000*GPU_RATE*total/len(rows),
            'gpu_usd_per_1000_judgments':1000*GPU_RATE*total/(6*len(rows)),
            'peak_allocated_gib':max(r['peak_allocated_gib'] for r in batches),
            'peak_reserved_gib':max(r['peak_reserved_gib'] for r in batches),
            'decisions_different_from_batch1': (sum(v!=baseline[(r['experiment'],r['id'])][k] for r in rows for k,v in r['values'].items()) if baseline else None),
            'code_label_correct_out_of_48':sum(v==r['labels'][k] for r in retrieval for k,v in r['values'].items()),
            'retrieval_top1_out_of_6':sum(i==1 for i in ranks),
            'support_agreement_out_of_144':sum(v==(r['reference_p_true'][k]>=.5) for r in support for k,v in r['values'].items())})
    return summaries


@app.local_entrypoint()
def main():
    from lab_benchmark import load_cases
    cases, references, manifest=load_cases()
    start=time.perf_counter();result=sweep.remote(cases)
    result['client_remote_seconds']=time.perf_counter()-start
    result['recorded_at']=datetime.now(timezone.utc).isoformat()
    baseline_path=ROOT/'results/every_lab_20260916T145029Z.json'
    result['historical_unbatched_records'] = ([r for r in json.loads(baseline_path.read_text())['records'] if r['steps']==1] if baseline_path.exists() else [])
    result['summary']=summarize(result)
    result['gpu_usd_per_second']=GPU_RATE
    result['pricing_source']='https://modal.com/pricing (checked 2026-09-16)'
    result['estimated_remote_function_gpu_usd']=GPU_RATE*result['remote_function_seconds']
    result['actual_invoice_cost_usd']=None
    result['excluded_prior_attempts']=['ap-TLGhmDV5HQsYKMVZlC3nQn: import failure', 'ap-znKziHAAFBScxjgRvenfMl: ascending sweep stopped on user request']
    result['artifact_sha256']=manifest['artifact_sha256']
    path=ROOT/'results'/('batch_sweep_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'path':str(path),'summary':result['summary'],'failures':result['failures'],
                      'resident_gib':result['resident_allocated_gib_after_setup'],
                      'estimated_run_gpu_usd':result['estimated_remote_function_gpu_usd']},indent=2))
