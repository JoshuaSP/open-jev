"""Render measured parallel runs alongside the original question-wise baseline."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base_path = root / "results/public_evals_20260916T193219Z.json"
base = json.loads(base_path.read_text())
paths = sorted((root / "results").glob("parallel_evals_*.json"))
runs = [(p, json.loads(p.read_text())) for p in paths]
lines = [
    "# Parallel JSON public evaluation",
    "",
    "All runs evaluate the same 408 questions from 46 saved Jev nodes across all 20 public cases. The accuracy denominator is the same 337 questions with unambiguous model-derived references. This replays Jev’s recorded paths; it does not reproduce autonomous branching or the full unpublished corpus.",
    "",
    "## Measured comparison",
    "",
    "| Runner | Steps | Accuracy | Inference time | Inference GPU cost | Input tokens |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for name, r in [("One question per sequence", base)] + [
    ("Grouped JSON", r) for _, r in runs
]:
    s = r["summary"]["ALL"]
    t = r["inference_totals"]
    lines.append(
        f"| {name} | {r['steps']} | {s['ours_correct']}/{s['scored']} ({s['ours_correct'] / s['scored']:.1%}) | {t['seconds']:.2f} s | ${t['gpu_usd']:.5f} | {t['input_tokens']:,} |"
    )
lines += [
    "",
    "Saved Jev scores 306/337 (90.8%) against these same references. Every grouped run preserves 408/408 valid typed answers.",
    "",
    "| Workflow | Original one-step | "
    + " | ".join(f"Grouped {r['steps']}-step" for _, r in runs)
    + " | Jev |",
    "| --- | ---: | " + " | ".join("---:" for _ in runs) + " | ---: |",
]
for w in [
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
]:
    s = base["summary"][w]
    n = s["scored"]
    cells = (
        [f"{s['ours_correct']}/{n}"]
        + [f"{r['summary'][w]['ours_correct']}/{n}" for _, r in runs]
        + [f"{s['jev_correct']}/{n}"]
    )
    lines.append("| " + w + " | " + " | ".join(cells) + " |")
lines += [
    "",
    "## Timing boundaries and cost",
    "",
    "Request time includes tokenization, field-layout construction, GPU execution, final readout and JSON decoding. No separate warmup is excluded. GPU estimates use $0.001097/second ($3.9492/hour), excluding CPU/RAM/storage and lifecycle outside the measured function. These are single exploratory runs, not latency percentiles or invoice totals.",
    "",
    "| Run | GPU function incl. setup and OOMs | GPU cost incl. setup and OOMs | Client wall time |",
    "| --- | ---: | ---: | ---: |",
]
for name, r in [("Original", base)] + [
    (f"Grouped {r['steps']}-step", r) for _, r in runs
]:
    lines.append(
        f"| {name} | {r['remote_function_seconds']:.2f} s | ${r['estimated_remote_function_gpu_usd']:.4f} | {r['client_seconds']:.2f} s |"
    )
for p, r in runs:
    t = r["inference_totals"]
    s = r["summary"]["ALL"]
    lines += [
        "",
        f"Grouped {r['steps']}-step: **{base['inference_totals']['seconds'] / t['seconds']:.2f}× faster** successful inference; **{t['seconds'] * 1000 / 408:.1f} ms amortized per judgment**, not individual request latency. GPU cost per million actual input tokens is ${t['gpu_usd_per_million_input_tokens']:.4f}. Input-token unit cost need not fall when repeated context is removed; cost for the fixed set of judgments is the relevant comparison.",
        f"Batch target starts at 64 and halves on OOM. Failed targets: {[f['target'] for f in r['failures']]}. Peak allocated memory in successful batches: {max(b['peak_allocated_gib'] for b in r['records']):.2f} GiB.",
    ]
lines += [
    "",
    "## What changed",
    "",
    "- Questions sharing a workflow node/document are packed into a JSON object using their original field names, instructions, criteria and allowed values.",
    "- Per-field candidate tries avoid enumerating the Cartesian product. Shorter alternatives are padded with JSON whitespace to keep subsequent fields aligned. We verify every field alternative decodes correctly.",
    "- All variable slots denoise over the full vocabulary; allowed alternatives are selected only from final logits. Shared JSON structure remains fixed. Multiple fields share attention, so judgment independence is not guaranteed.",
    "- The fixed 256-token output canvas limits object size. 41 nodes fit in one canvas; five larger nodes split. Total: 56 canvases, merged into 46 complete node JSON objects. The large nodes still repeat context between their chunks; there is no cross-canvas KV reuse.",
    "- Different schemas can share a batch. Jobs are sorted by actual prompt length, and the batch target halves from 64 only on OOM. No repeated cases are added to fill batches.",
    "- Grouping changes prompts, token layout, padding and attention interactions. The accuracy changes are measured, not assumed away. This is not an isolated kernel benchmark.",
    "",
    "## Reproduce",
    "",
    "```bash",
    "modal run verify_parallel.py  # CPU constraint checks",
    "modal run parallel_benchmark.py --steps 1",
    "modal run parallel_benchmark.py --steps 2",
    "python scripts/report_parallel.py",
    "```",
    "",
    "## Receipts",
    "",
]
for p, r in runs:
    lines.append(
        f"- [{r['steps']}-step grouped receipt]({p.relative_to(root)}) — includes each emitted canvas and `node_outputs`, the complete merged JSON objects."
    )
lines += ["- [Original baseline](results/public_evals_20260916T193219Z.json)", ""]
(root / "PARALLEL_EVALS.md").write_text("\n".join(lines))
print(root / "PARALLEL_EVALS.md")
