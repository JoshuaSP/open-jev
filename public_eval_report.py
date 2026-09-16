"""Audit counts and write the public evaluation report without another GPU run."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    path = ROOT / 'results/public_evals_20260916T193219Z.json'
    result = json.loads(path.read_text())
    audit = json.loads((ROOT / 'results/public_eval_token_audit.json').read_text())
    counts = audit['counts']
    assert set(counts) == {r['id'] for r in result['predictions']}
    for row in result['predictions']:
        row['input_tokens'] = counts[row['id']]
    offset = 0
    for batch in result['records']:
        rows = result['predictions'][offset:offset+batch['batch_size']]
        batch['question_ids'] = [r['id'] for r in rows]
        batch['input_tokens'] = sum(r['input_tokens'] for r in rows)
        batch['max_input_tokens'] = max(r['input_tokens'] for r in rows)
        assert batch['padded_input_tokens'] >= batch['input_tokens']
        offset += batch['batch_size']
    assert offset == len(result['predictions'])
    total = result['inference_totals']
    total['input_tokens'] = sum(counts.values())
    total['gpu_usd_per_million_input_tokens'] = total['gpu_usd'] * 1e6 / total['input_tokens']
    result['token_audit'] = dict(source='results/public_eval_token_audit.json',
        note='Counts independently recomputed on CPU from attention_mask using identical prompts, processor and model revision. Original len(input_ids) counted the batch dimension, so the first run retained source order within each answer schema instead of sorting by length. Batch membership is now recorded explicitly; runner corrected for future runs.')
    by_type = collections.defaultdict(collections.Counter)
    for r in result['predictions']:
        if r['reference_label'] is None or r['jev_label'] is None:
            continue
        value = str(r['value']).lower() if isinstance(r['value'], bool) else str(r['value'])
        t = by_type[r['type']]
        t['scored'] += 1
        t['ours_correct'] += int(value == r['reference_label'])
        t['jev_correct'] += int(r['jev_label'] == r['reference_label'])
    result['summary_by_type'] = {k: dict(v) for k, v in by_type.items()}
    result['modal_app_id'] = 'ap-izROZ1hzl5ir87j2UfUt9W'
    path.write_text(json.dumps(result, indent=2)+'\n')

    lines = ['# Public TypeSafe examples: one-step DiffusionGemma vs saved Jev', '',
             'Run date: 2026-09-16. This is a question-level replay of the nodes Jev ran on 20 selected public examples, not a reproduction of independent workflow branching or their full 711-case evaluation.', '',
             '## Accuracy', '', '| Workflow | Scored questions | Ours | Jev |', '| --- | ---: | ---: | ---: |']
    for name, s in result['summary'].items():
        n = s['scored']
        lines.append(f'| {name} | {n} | {s["ours_correct"]}/{n} ({s["ours_correct"]/n:.1%}) | {s["jev_correct"]}/{n} ({s["jev_correct"]/n:.1%}) |')
    s = result['summary']['ALL']
    lines += ['', f'All {s["questions_run"]} answers were evaluated for output validity; {s["valid"]} were valid. We exclude {s.get("missing_reference",0)} questions without references and {s.get("reference_tie",0)} tied reference answers from accuracy. References are model-derived, not human ground truth.', '',
              '| Answer type | Scored | Ours correct | Jev correct |', '| --- | ---: | ---: | ---: |']
    for typ, s in result['summary_by_type'].items():
        lines.append(f'| {typ} | {s["scored"]} | {s["ours_correct"]} | {s["jev_correct"]} |')
    lines += ['', '## Runtime and cost', '',
              f'- Successful inference requests: **{total["seconds"]:.2f} seconds**, **${total["gpu_usd"]:.4f} GPU cost**. Includes first-request overhead; no separate warmup is excluded.',
              f'- GPU function including setup, OOM attempts and cleanup: **{result["remote_function_seconds"]:.2f} seconds**, **${result["estimated_remote_function_gpu_usd"]:.4f}**.',
              f'- Client wall time: **{result["client_seconds"]:.2f} seconds**.',
              f'- Actual prompt tokens: **{total["input_tokens"]:,}** including templates, instructions and repeated documents. Successful inference cost: **${total["gpu_usd_per_million_input_tokens"]:.4f}/million input tokens**.',
              '- H100 rate: $0.001097/second ($3.9492/hour), GPU-only estimate. Excludes CPU/RAM/storage, the CPU token audit, and lifecycle outside the function. Not an invoice total.',
              f'- Batch target started at 64; caught OOM targets: {[f["requested_batch_size"] for f in result["failures"]]}. Actual successful batch sizes: {sorted({b["batch_size"] for b in result["records"]})}.',
              f'- Peak allocated CUDA memory: {max(b["peak_allocated_gib"] for b in result["records"]):.2f} GiB; peak reserved: {max(b["peak_reserved_gib"] for b in result["records"]):.2f} GiB.', '',
              f'For context, Jev\'s saved whole-workflow records sum to {sum(c["seconds"] for c in result["jev_saved_case_costs"]):.2f} seconds and ${sum(c["cost"]["usd"] for c in result["jev_saved_case_costs"]):.6f} estimated API cost. This is not an equivalent speed/cost measurement: our runner repeats the document once per question, while their workflow groups questions. We did not call the Jev API.', '',
              '## Method', '',
              '- Original public documents, question wording and criteria. Each request asks for a literal JSON object containing `answer`, with the original boolean, score or named choice; no letter-label indirection.',
              '- One denoising step, seed 0 full-vocabulary noise, fixed shared structural tokens. Final logits select the first differing allowed token; later branching positions discriminate remaining candidates and unique suffixes are forced. This is greedy candidate selection, not sequence likelihood scoring.',
              '- Saved Jev model: `typesafe:v13_snowy_elephant`. Boolean labels use its probability argmax; choices and scores use the mode of the saved distribution.',
              '- Reference distributions are averaged exactly as in the public viewer; missing distributions become one-hot votes. Reference ties are excluded. Numeric scores are evaluated as category argmax, not expected-value error.',
              '- The inference worker receives no saved answers or references. Node selection and follow-up document context come from the saved Jev path, so this does not test whether our model would open the right branches.',
              '- Public cases were selected to illustrate disagreements and agreements. These percentages do not estimate performance on the undisclosed corpus.',
              '- Token counts were audited and corrected on CPU. The first GPU run mistakenly used the batch dimension for length sorting; predictions and timings are unchanged. The runner now counts attention-mask tokens correctly. Original batch membership is preserved in the receipt.', '',
              '## Artifacts', '', f'- [Full receipt and individual predictions]({path.relative_to(ROOT)})',
              '- [Runner](public_eval_benchmark.py)', '- [Token audit](public_eval_token_audit.py)',
              '- [Source data provenance](typesafe-public-evals/README.md)',
              '- [TypeSafe public evaluations](https://evals.typesafe.ai/)', '']
    (ROOT / 'PUBLIC_EVALS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=result['summary'], types=result['summary_by_type'], inference=total,
                         function_seconds=result['remote_function_seconds'], gpu_cost=result['estimated_remote_function_gpu_usd']), indent=2))


if __name__ == '__main__':
    main()
