"""Replay Jev's public question nodes: modal run public_eval_benchmark.py.

Question-level evaluation, not a reproduction of the policy/branching benchmark.
References and Jev outputs stay on the client and are never included in prompts.
"""
import collections
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
from modal_app import MODEL, cache, hf_auth, image, secret

ROOT = Path(__file__).parent
GPU_RATE = 0.001097
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
app = modal.App('open-jev-public-evals')
image = image.add_local_python_source('modal_app')


def canonical(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def load_cases():
    inputs, labels, sources, saved_costs = [], {}, {}, []
    for path in sorted((ROOT / 'typesafe-public-evals').glob('*.json')):
        if path.name == 'manifest.json':
            continue
        data = json.loads(path.read_text())
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for case_id, case in data['cases'].items():
            jev = case['models']['typesafe']
            saved_costs.append(dict(workflow=path.stem, case=case_id,
                                    model=jev['model'], cost=jev.get('cost'), seconds=jev.get('seconds')))
            for node in jev['nodes']:
                if not node['ran']:
                    continue
                for qid, index in node['questions'].items():
                    question = data['questions'][index]
                    typ = question['type']
                    options = ([False, True] if typ == 'noul' else
                               list(range(len(question['criteria']))) if typ == 'score' else
                               list(question['criteria']))
                    identity = '/'.join((path.stem, case_id, node['node'], qid))
                    instructions = (question['instructions'] + '\n\nCriteria: ' +
                                    json.dumps(question['criteria'], ensure_ascii=False) +
                                    '\nAllowed answer values: ' + json.dumps(options, ensure_ascii=False) +
                                    '\nReturn exactly one JSON object with the key "answer" and the chosen value. '
                                    'For numeric scores, criteria are indexed from 0. No explanation.')
                    document = json.dumps(data['documents'][node['doc']], ensure_ascii=False)
                    prompt = '<user_text>\n' + document + '\n</user_text>\n\n<instructions>\n' + instructions + '\n</instructions>'
                    inputs.append(dict(id=identity, workflow=path.stem, case=case_id,
                                       type=typ, options=options, prompt=prompt))
                    reference = case['reference_answers'].get(node['node'], {}).get(qid)
                    answer = node['answers'][qid]
                    probs = ({'false': 1-answer['noul'], 'true': answer['noul']}
                             if typ == 'noul' else answer.get('probabilities'))
                    reported = ((answer['noul'] >= .5) if typ == 'noul' else
                                answer.get('choice') if typ == 'choice' else None)
                    labels[identity] = dict(reference=reference, jev=answer,
                                           jev_probabilities=probs, jev_reported=reported)
    assert len({x['id'] for x in inputs}) == len(inputs)
    return inputs, labels, sources, saved_costs


@app.function(image=image, secrets=[secret], volumes={'/cache': cache}, gpu='H100', timeout=1800)
def evaluate(rows):
    started = time.perf_counter()
    import gc
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, LogitsProcessorList
    from json_canvas import FinalTrieCanvas, constrained_sampler

    token = hf_auth()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, token=token)
    processor.tokenizer.padding_side = 'left'
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, revision=REVISION, token=token, dtype=torch.bfloat16, device_map='cuda').eval()
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    groups = collections.defaultdict(list)
    for row in rows:
        encoded = processor.apply_chat_template(
            [{'role': 'user', 'content': row['prompt']}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt')
        row['input_tokens'] = int(encoded['attention_mask'].sum())
        groups[json.dumps(row['options'])].append(row)
    setup = time.perf_counter() - started
    predictions, records, failures = [], [], []
    max_batch = 64
    for options_json, group in groups.items():
        options = json.loads(options_json)
        layout = FinalTrieCanvas(processor.tokenizer, [json.dumps({'answer': v}) for v in options],
                                 model.config.canvas_length, model.config.text_config.vocab_size, eos, model.device)
        torch.manual_seed(0)
        initial = layout.initialize_canvas(1, model.device)
        layout.initialize_canvas = lambda batch_size, device, initial=initial: initial.expand(batch_size, -1).clone()
        group.sort(key=lambda x: x['input_tokens'])
        offset = 0
        while offset < len(group):
            chunk = group[offset:offset + max_batch]
            def run():
                start = time.perf_counter()
                inputs = processor.apply_chat_template(
                    [[{'role': 'user', 'content': r['prompt']}] for r in chunk], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt',
                    processor_kwargs={'padding': True}).to(model.device)
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode(), constrained_sampler(model, layout):
                    torch.cuda.synchronize()
                    gen_start = time.perf_counter()
                    model.generate(**inputs, max_new_tokens=model.config.canvas_length, max_denoising_steps=1,
                                   cache_implementation='dynamic', logits_processor=LogitsProcessorList([layout]))
                    torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - gen_start
                values = [layout.decode(ids)[1]['answer'] for ids in layout.select_final().tolist()]
                return values, dict(request_seconds=time.perf_counter()-start,
                                    generation_seconds=generation_seconds, batch_size=len(chunk),
                                    requested_batch_size=max_batch,
                                    input_tokens=sum(r['input_tokens'] for r in chunk),
                                    padded_input_tokens=int(inputs['input_ids'].numel()),
                                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            try:
                values, stat = run()
            except torch.cuda.OutOfMemoryError as error:
                failures.append(dict(batch_size=len(chunk), requested_batch_size=max_batch,
                                     max_input_tokens=max(r['input_tokens'] for r in chunk), error=str(error)))
                layout.last_value_logits = None
                gc.collect()
                torch.cuda.empty_cache()
                if max_batch == 1:
                    raise
                max_batch //= 2
                print(f'OOM; reducing batch target to {max_batch}', flush=True)
                continue
            records.append(stat)
            for row, value in zip(chunk, values):
                predictions.append(dict(id=row['id'], workflow=row['workflow'], case=row['case'],
                                        type=row['type'], value=value, input_tokens=row['input_tokens'],
                                        valid=value in row['options']))
            offset += len(chunk)
            print(f'Completed {len(predictions)}/{len(rows)} questions; batch {len(chunk)}, {stat["request_seconds"]:.2f}s', flush=True)
        del layout, initial
        gc.collect()
        torch.cuda.empty_cache()
    return dict(model=MODEL, revision=REVISION, steps=1, seed=0, predictions=predictions,
                records=records, failures=failures, setup_seconds=setup,
                remote_function_seconds=time.perf_counter()-started,
                torch=str(torch.__version__), transformers=transformers.__version__,
                hardware=torch.cuda.get_device_name(),
                selection='Final raw-logit greedy candidate trie; first differing token discriminates, unique suffix forced. No answer restrictions during denoising.')


def winner(probabilities):
    if not probabilities:
        return None
    maximum = max(probabilities.values())
    winners = [k for k, v in probabilities.items() if abs(v-maximum) < 1e-9]
    return winners[0] if len(winners) == 1 else None


def summarize(result, labels):
    totals = collections.defaultdict(collections.Counter)
    for row in result['predictions']:
        label = labels[row['id']]
        reference = label['reference']
        consensus = collections.defaultdict(float)
        if reference:
            for item in reference['sets']:
                probs = item.get('probabilities') or {canonical(item['value']): 1.0}
                for key, value in probs.items():
                    consensus[key] += value / len(reference['sets'])
        target = winner(consensus)
        jev_value = winner(label['jev_probabilities'])
        if jev_value is None and not label['jev_probabilities'] and label['jev_reported'] is not None:
            jev_value = canonical(label['jev_reported'])
        row.update(reference_distribution=dict(consensus), reference_label=target, jev_label=jev_value,
                   jev_raw=label['jev'], jev_reported=label['jev_reported'])
        for name in (row['workflow'], 'ALL'):
            t = totals[name]
            t['questions_run'] += 1
            t['valid'] += int(row['valid'])
            if target is None:
                t['missing_reference' if not reference else 'reference_tie'] += 1
                continue
            if jev_value is None:
                t['jev_tie_or_missing'] += 1
                continue
            ours = canonical(row['value'])
            t['scored'] += 1
            t['ours_correct'] += int(ours == target)
            t['jev_correct'] += int(jev_value == target)
            t['ours_jev_agree'] += int(ours == jev_value)
            t['ours_only_correct'] += int(ours == target and jev_value != target)
            t['jev_only_correct'] += int(jev_value == target and ours != target)
    return {k: dict(v) for k, v in totals.items()}


@app.local_entrypoint()
def main():
    rows, labels, sources, saved_costs = load_cases()
    print(f'Replaying {len(rows)} questions across {len(saved_costs)} public cases. References remain local.')
    start = time.perf_counter()
    result = evaluate.remote(rows)
    result.update(client_seconds=time.perf_counter()-start, recorded_at=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sources, jev_saved_case_costs=saved_costs, gpu_usd_per_second=GPU_RATE,
                  cost_scope='GPU-only estimate at $3.9492/hour; excludes CPU/RAM/storage and lifecycle outside function',
                  scope='Question replay on nodes Jev ran, not independent branching or final policy decisions; selected public examples only.')
    result['summary'] = summarize(result, labels)
    result['estimated_remote_function_gpu_usd'] = GPU_RATE * result['remote_function_seconds']
    measured = sum(r['request_seconds'] for r in result['records'])
    tokens = sum(r['input_tokens'] for r in result['records'])
    result['inference_totals'] = dict(seconds=measured, input_tokens=tokens,
                                    gpu_usd=GPU_RATE*measured, gpu_usd_per_million_input_tokens=GPU_RATE*measured*1e6/tokens)
    path = ROOT / 'results' / ('public_evals_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), summary=result['summary'], inference=result['inference_totals'],
                          run_gpu_usd=result['estimated_remote_function_gpu_usd']), indent=2))
