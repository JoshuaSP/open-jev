# Parallel JSON public evaluation

All runs evaluate the same 408 questions from 46 saved Jev nodes across all 20 public cases. The accuracy denominator is the same 337 questions with unambiguous model-derived references. This replays Jev’s recorded paths; it does not reproduce autonomous branching or the full unpublished corpus.

## Measured comparison

| Runner | Steps | Accuracy | Inference time | Inference GPU cost | Input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| One question per sequence | 1 | 298/337 (88.4%) | 248.03 s | $0.27209 | 2,214,128 |
| Grouped JSON | 1 | 290/337 (86.1%) | 48.61 s | $0.05332 | 300,431 |
| Grouped JSON | 2 | 296/337 (87.8%) | 51.11 s | $0.05607 | 300,431 |

Saved Jev scores 306/337 (90.8%) against these same references. Every grouped run preserves 408/408 valid typed answers.

| Workflow | Original one-step | Grouped 1-step | Grouped 2-step | Jev |
| --- | ---: | ---: | ---: | ---: |
| agent_trace_observability | 41/52 | 41/52 | 42/52 | 41/52 |
| customer_service | 84/92 | 76/92 | 78/92 | 82/92 |
| invoice_processing | 155/167 | 154/167 | 157/167 | 162/167 |
| security_incidents | 18/26 | 19/26 | 19/26 | 21/26 |

## Timing boundaries and cost

Request time includes tokenization, field-layout construction, GPU execution, final readout and JSON decoding. No separate warmup is excluded. GPU estimates use $0.001097/second ($3.9492/hour), excluding CPU/RAM/storage and lifecycle outside the measured function. These are single exploratory runs, not latency percentiles or invoice totals.

| Run | GPU function incl. setup and OOMs | GPU cost incl. setup and OOMs | Client wall time |
| --- | ---: | ---: | ---: |
| Original | 339.07 s | $0.3720 | 348.32 s |
| Grouped 1-step | 180.32 s | $0.1978 | 193.34 s |
| Grouped 2-step | 148.97 s | $0.1634 | 155.00 s |

Grouped 1-step: **5.10× faster** successful inference; **119.1 ms amortized per judgment**, not individual request latency. GPU cost per million actual input tokens is $0.1775. Input-token unit cost need not fall when repeated context is removed; cost for the fixed set of judgments is the relevant comparison.
Batch target starts at 64 and halves on OOM. Failed targets: [64, 32, 16, 8, 4]. Peak allocated memory in successful batches: 71.79 GiB.

Grouped 2-step: **4.85× faster** successful inference; **125.3 ms amortized per judgment**, not individual request latency. GPU cost per million actual input tokens is $0.1866. Input-token unit cost need not fall when repeated context is removed; cost for the fixed set of judgments is the relevant comparison.
Batch target starts at 64 and halves on OOM. Failed targets: [64, 32, 16, 8, 4]. Peak allocated memory in successful batches: 71.79 GiB.

## What changed

- Questions sharing a workflow node/document are packed into a JSON object using their original field names, instructions, criteria and allowed values.
- Per-field candidate tries avoid enumerating the Cartesian product. Shorter alternatives are padded with JSON whitespace to keep subsequent fields aligned. We verify every field alternative decodes correctly.
- All variable slots denoise over the full vocabulary; allowed alternatives are selected only from final logits. Shared JSON structure remains fixed. Multiple fields share attention, so judgment independence is not guaranteed.
- The fixed 256-token output canvas limits object size. 41 nodes fit in one canvas; five larger nodes split. Total: 56 canvases, merged into 46 complete node JSON objects. The large nodes still repeat context between their chunks; there is no cross-canvas KV reuse.
- Different schemas can share a batch. Jobs are sorted by actual prompt length, and the batch target halves from 64 only on OOM. No repeated cases are added to fill batches.
- Grouping changes prompts, token layout, padding and attention interactions. The accuracy changes are measured, not assumed away. This is not an isolated kernel benchmark.

## Reproduce

```bash
modal run verify_parallel.py  # CPU constraint checks
modal run parallel_benchmark.py --steps 1
modal run parallel_benchmark.py --steps 2
python scripts/report_parallel.py
```

## Receipts

- [1-step grouped receipt](results/parallel_evals_20260916T221434Z.json) — includes each emitted canvas and `node_outputs`, the complete merged JSON objects.
- [2-step grouped receipt](results/parallel_evals_20260916T221744Z.json) — includes each emitted canvas and `node_outputs`, the complete merged JSON objects.
- [Original baseline](results/public_evals_20260916T193219Z.json)
