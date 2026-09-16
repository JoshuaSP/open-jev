# One-step batching results

Run: `modal run batch_benchmark.py`
Receipt: `results/batch_sweep_20260916T145918Z.json`

The run started at 64 documents, then halved only on CUDA OOM:

| Batch | Outcome |
| --- | --- |
| 64 | OOM |
| 32 | OOM |
| 16 | Fits; three measured passes completed |

One H100 80GB, BF16, one denoising step, six boolean decisions per document.
Batch 16 was the first successful size in this halving search, not a proven
maximum or cost optimum. No smaller-size sweep was run.

| Measurement | Result |
| --- | --- |
| Throughput | 30.41 documents/sec (182.44 judgments/sec) |
| Median full-batch request latency | 531 ms |
| Amortized time per document | 32.89 ms |
| Estimated GPU cost / 1,000 documents | $0.03608 |
| Estimated GPU cost / 1,000 judgments | $0.006013 |
| Resident PyTorch allocation after setup | 48.23 GiB |
| Peak PyTorch allocation during measured batches | 77.34 GiB |
| Peak PyTorch reserved memory | 78.03 GiB |
| Device capacity reported by CUDA | 79.18 GiB |

Memory figures are CUDA/PyTorch measurements, not a component-by-component memory
profile. The resident allocation includes weights and our two canvas layouts;
peak usage additionally includes attention/KV/activation/logit/sampler buffers.
Reserved memory includes allocator-held memory, and is not additional to allocated
memory. There is little headroom at batch 16 in this implementation.

The workload contains only 8 unique code documents and 24 support messages. Each
experiment was cycled into 64 document slots, with three measured passes over both
experiments (384 document evaluations / 2,304 judgments). These are repeated-input
throughput measurements, not 384 independent accuracy cases. The model still runs
a separate sequence per slot; prompt KV caches are not reused between calls.

Questions sharing a schema are batched together. All sizes use left-padding to a
fixed maximum prompt length per experiment, and every document starts with the
same seed-0 full-vocabulary noise row. Warmups cover each tested schema/batch shape
and are excluded from steady-state throughput. End-to-end in-container request
time includes tokenization, transfer, generation, final readout, and parsing;
network latency and batch-queue waiting are excluded.

## Quality

On the 32 unique documents:

- Code retrieval: 48/48 relevance labels correct; all 6 queries rank the correct
  file first.
- Support: 138/144 decisions agree with saved TypeSafe probabilities thresholded
  at 0.5, unchanged in aggregate from the historical one-step unbatched run.
- All repeated copies and passes yield consistent decisions for each document.
- Two unique support decisions changed versus that historical run: ticket-10
  billing_issue became true (TypeSafe 0.53), and ticket-18 churn_risk became true
  (TypeSafe 0.40). Across repeats these account for 15 changed decision instances.
  The historical baseline used different padding, so this does not isolate a
  batching-only numerical effect. Exact logits were not compared.

## Cost accounting

Steady-state GPU estimates use $0.001097/H100-second from
[Modal pricing](https://modal.com/pricing), checked 2026-09-16. The measured remote
function's GPU estimate is $0.12375 including loading, warmups, and OOM attempts.
It excludes CPU, RAM, storage, lifecycle time outside the function, and prior
aborted/import-failed apps. It is not an invoice total. Actual billed cost is unknown.
Do not extrapolate the steady-state price to sparse traffic or cold-start-heavy
usage: it assumes batches are available to keep the GPU occupied.

## Input-token pricing

The 32 unique prompts contain 6,534 non-padding input tokens (2,274 code retrieval; 4,260 customer support), including instructions and chat formatting. The measured repeated workload contains 88,725 non-padding tokens, or 95,616 including padding. Its $0.01385395 steady-state GPU cost corresponds to **$0.1561 per million non-padding input tokens**, including the inference work to produce the decisions. Token counts are joined by document ID from the earlier receipt, whose prompts and tokenizer match this run. This is workload-specific GPU cost, not a hosted API price; cold starts, CPU, RAM, storage, and queueing remain excluded.
