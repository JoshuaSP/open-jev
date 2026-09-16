"""Run Every's two boolean experiments: modal run lab_benchmark.py"""
import hashlib
import itertools
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
from modal_app import image, secret, cache, MODEL, REVISION, hf_auth

app = modal.App("open-jev-every-lab")
image = image.add_local_python_source("modal_app")

ROOT = Path(__file__).parent
LAB = ROOT / 'typesafe-lab'
GPU_RATE = 0.001097  # USD/H100-second, modal.com/pricing, checked 2026-09-16


def load_cases():
    if not (LAB / 'experiments/RUN_MANIFEST.json').exists():
        raise FileNotFoundError('Import the Every bundle first: python scripts/import_every.py /path/to/typesafe-lab.zip')
    manifest = json.loads((LAB / 'experiments/RUN_MANIFEST.json').read_text())
    for name, expected in manifest['artifact_sha256'].items():
        actual = hashlib.sha256((LAB / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Bundle hash mismatch: {name}')
    data = json.loads((LAB / 'public/experiments.json').read_text())
    cases, references = [], {}
    for e in data['experiments']:
        if e['id'] == 'code-rag':
            questions = {q['id']: q['question'] for q in e['grid']}
            for doc in e['documents']:
                cells = {q['id']: next(c for c in q['cells'] if c['document'] == doc['id']) for q in e['grid']}
                cases.append({'experiment': e['id'], 'id': doc['id'], 'questions': questions,
                              'document': (LAB / 'fixtures/code-repo' / doc['id']).read_text(),
                              'reference': {k: c['value'] for k, c in cells.items()},
                              'labels': {k: c['relevant'] for k, c in cells.items()}})
        elif e['id'] == 'customer-voice':
            for row in e['rows']:
                cases.append({'experiment': e['id'], 'id': row['id'], 'questions': e['questions'],
                              'document': row['document'], 'reference': row['signals'], 'labels': None})
        else:
            continue
        references[e['id']] = {'measured': e['measured'], 'metrics': e['metrics']}
    assert len(cases) == 32 and sum(len(c['questions']) for c in cases) == 192
    return cases, references, manifest


@app.function(image=image, secrets=[secret], volumes={'/cache': cache}, gpu='H100', timeout=900)
def benchmark(cases):
    started = time.perf_counter()
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, LogitsProcessorList
    from json_canvas import FinalReadoutCanvas, constrained_sampler

    token = hf_auth()
    revision = REVISION
    processor = AutoProcessor.from_pretrained(MODEL, revision=revision, token=token)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, revision=revision, token=token, dtype=torch.bfloat16, device_map='cuda').eval()
    cache.commit()
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    layouts = {}
    for case in cases:
        if case['experiment'] in layouts:
            continue
        keys = list(case['questions'])
        if len(keys) > 8:
            raise ValueError('This finite-language prototype supports at most eight boolean fields')
        texts = [json.dumps(dict(zip(keys, values))) for values in itertools.product((False, True), repeat=len(keys))]
        layout = FinalReadoutCanvas(processor.tokenizer, texts, model.config.canvas_length,
                                   model.config.text_config.vocab_size, eos, model.device)
        assert len(layout.variable_positions) == len(keys)
        # Verify exactly one independent value position per question, in field order.
        base = processor.tokenizer.encode(texts[0], add_special_tokens=False)
        for key, position in zip(keys, layout.variable_positions):
            obj = dict.fromkeys(keys, False); obj[key] = True
            alternate = processor.tokenizer.encode(json.dumps(obj), add_special_tokens=False)
            assert len(base) == len(alternate)
            assert [i for i, (a, b) in enumerate(zip(base, alternate)) if a != b] == [position]
        layouts[case['experiment']] = layout
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started

    def run(case, steps):
        request_start = time.perf_counter()
        layout = layouts[case['experiment']]
        layout.steps = 0
        instructions = ('For each question below, return a boolean answer in a JSON object. '
                        'Use exactly the listed keys, in this order. No explanations.\n'
                        + '\n'.join(json.dumps(k) + ': ' + q for k, q in case['questions'].items()))
        prompt = '<user_text>\n' + case['document'] + '\n</user_text>\n\n<instructions>\n' + instructions + '\n</instructions>'
        inputs = processor.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
        torch.manual_seed(0)
        with constrained_sampler(model, layout):
            torch.cuda.synchronize()
            gen_start = time.perf_counter()
            model.generate(**inputs, max_new_tokens=model.config.canvas_length, max_denoising_steps=steps,
                           cache_implementation='dynamic', logits_processor=LogitsProcessorList([layout]))
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - gen_start
        final_ids = layout.select_final()[0].tolist()
        text, values = layout.decode(final_ids)
        p_true = {}
        for key, pos, scores in zip(case['questions'], layout.variable_positions, layout.last_value_logits):
            true_index = next(i for i, t in enumerate(layout.columns[pos])
                              if processor.tokenizer.decode([t]).strip() == 'true')
            p_true[key] = scores.softmax(-1)[0, true_index].item()
        request_seconds = time.perf_counter() - request_start
        return {'experiment': case['experiment'], 'id': case['id'], 'steps': steps, 'seed': 0,
                'actual_steps': layout.steps, 'input_tokens': inputs['input_ids'].shape[1],
                'judgments': len(values), 'json': text, 'values': values,
                'uncalibrated_p_true': p_true, 'reference_p_true': case['reference'], 'labels': case['labels'],
                'generation_seconds': generation_seconds, 'request_seconds': request_seconds,
                'estimated_gpu_cost_usd': request_seconds * GPU_RATE, 'prompt': prompt}

    warmup = run(cases[0], 1)
    records = []
    loop_start = time.perf_counter()
    for steps in (1, 4):
        for case in cases:
            records.append(run(case, steps))
        print(f'Completed {len(cases)} documents at {steps} steps', flush=True)
    return {'model': MODEL, 'revision': revision, 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'hardware': torch.cuda.get_device_name(),
            'setup_seconds': setup_seconds, 'warmup': warmup,
            'benchmark_loop_seconds': time.perf_counter() - loop_start,
            'remote_function_seconds': time.perf_counter() - started, 'records': records}


def summarize(records):
    output = []
    for experiment in sorted({r['experiment'] for r in records}):
        for steps in sorted({r['steps'] for r in records}):
            rows = [r for r in records if r['experiment'] == experiment and r['steps'] == steps]
            pairs = [(r['values'][k], r['reference_p_true'][k]) for r in rows for k in r['values']]
            latencies = sorted(r['request_seconds'] for r in rows)
            summary = {'experiment': experiment, 'steps': steps, 'calls': len(rows), 'judgments': len(pairs),
                       'agreement_with_saved_typesafe_at_0_5': sum(v == (p >= .5) for v, p in pairs) / len(pairs),
                       'median_request_ms': statistics.median(latencies) * 1000,
                       'p95_request_ms_nearest_rank': latencies[__import__('math').ceil(.95 * len(rows)) - 1] * 1000,
                       'serial_request_seconds': sum(latencies),
                       'estimated_gpu_cost_usd': sum(r['estimated_gpu_cost_usd'] for r in rows)}
            if experiment == 'code-rag':
                ranks = []
                for key in rows[0]['values']:
                    ranked = sorted(rows, key=lambda r: (-r['uncalibrated_p_true'][key], r['id']))
                    ranks.append(next(i for i, r in enumerate(ranked, 1) if r['labels'][key]))
                summary.update(recall_at_1=sum(i == 1 for i in ranks)/len(ranks),
                               recall_at_3=sum(i <= 3 for i in ranks)/len(ranks),
                               mrr=sum(1/i for i in ranks)/len(ranks),
                               label_accuracy=sum(r['values'][k] == r['labels'][k] for r in rows for k in r['values'])/len(pairs))
            output.append(summary)
    return output


@app.local_entrypoint()
def main():
    cases, references, manifest = load_cases()
    client_start = time.perf_counter()
    result = benchmark.remote(cases)
    result['client_remote_call_seconds'] = time.perf_counter() - client_start
    result['recorded_at'] = datetime.now(timezone.utc).isoformat()
    result['source_zip_sha256'] = hashlib.sha256((ROOT / 'typesafe-lab-source.zip').read_bytes()).hexdigest()
    result['bundle_artifact_hashes'] = manifest['artifact_sha256']
    result['saved_typesafe_references'] = references
    result['summary'] = summarize(result['records'])
    result['cost_accounting'] = {
        'gpu_usd_per_second': GPU_RATE, 'rate_source': 'https://modal.com/pricing', 'rate_checked': '2026-09-16',
        'estimated_gpu_cost_remote_function_usd': result['remote_function_seconds'] * GPU_RATE,
        'actual_invoice_cost_usd': None,
        'prior_unmetered_attempts': [{'app_id': 'ap-OoaCnCReRIswwHh5Ug428N', 'reason': 'GPU evaluation completed but result deserialization failed on TorchVersion; cost unavailable, not included in this run estimate'}],
        'excluded': 'CPU, RAM, storage, image build, container startup before function entry and teardown; account discounts/credits',
        'timing_note': 'Per-request time includes tokenization, generation, final readout and parsing; calls are serial. Client remote-call time includes scheduling. Saved TypeSafe experiment wall times used concurrent calls; compare per-call times, not batch wall time.'}
    path = ROOT / 'results' / ('every_lab_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'result_file': str(path), 'summary': result['summary'], 'cost': result['cost_accounting']}, indent=2))
