# Every lab integration

The archive `typesafe-lab-source.zip` was copied from Downloads and extracted to
`typesafe-lab/`. The original notebook is
`typesafe-lab/notebooks/TypeSafe_Parallel_Judgment_Lab.ipynb`. The seven data hashes
in its manifest were verified. Original data, notebooks, and recorded TypeSafe
results are preserved.

Import your downloaded bundle with `python scripts/import_every.py /path/to/typesafe-lab.zip`. The original notebook remains in that local bundle.
Run `modal run lab_benchmark.py` from this project to produce a timestamped JSON
receipt under `results/`. No TypeSafe or other hosted model API is called by this
adapter; the recorded TypeSafe results are historical references from the bundle.

The first run covers two complete experiments at 1 and 4 denoising steps:

- Code retrieval: 8 full source documents × 6 exact questions = 48 judgments.
- Customer voice: 24 support messages × 6 exact questions = 144 judgments.

There are 64 document evaluations and 384 judgments across both budgets, plus
one warmup. Each document is evaluated once per budget with seed 0. The other
nine experiments have not been run. Choices, rubric scores, agent trajectories,
and the original downstream workflows need further adapter work.

Each document is bracketed as user text, followed by the original questions and
instructions requesting boolean JSON. Every field is denoised in parallel on one
canvas, then chosen by allowed-token argmax at final readout. Full-vocabulary
noise is used for answer slots; only JSON structure is fixed during denoising.
For Noul-style ranking, we expose uncalibrated two-token softmax scores. This does
not reproduce Jev's calibration or independent-question attention isolation.

Retrieval rankings use those scores and the original relevance labels. Customer
voice measures agreement with saved TypeSafe decisions at probability >= 0.5,
not ground-truth accuracy. Tie-breaking for retrieval is deterministic by document
ID. We record the exact prompts, model revision, package versions, predictions,
reference scores, and source hashes.

Per-request timings include tokenization, generation, final readout, and JSON
parsing. Generation time is also recorded separately with CUDA synchronization.
Setup/loading, warmup, total remote-function time, and client remote-call wall time
are separate fields. Calls are serial on one H100. Every's saved API batch wall
times used concurrent calls: compare per-call distributions, not batch wall time.

GPU costs use the published H100 rate of $0.001097/second, checked 2026-09-16 at
[Modal pricing](https://modal.com/pricing). Per-call estimates and the larger
remote-function estimate (including setup and warmup) are both retained.
They exclude CPU, RAM, storage, image build, lifecycle time outside the function,
and account discounts/credits. Actual invoice cost is unavailable and remains null.
One initial GPU run completed but its result transfer failed because TorchVersion
was not serializable into the local environment; its unknown cost is disclosed
separately and is not included in the successful-run estimate.

## First measured results

Receipt: `results/every_lab_20260916T145029Z.json`.

| Experiment | Steps | Result | Median request | Estimated GPU cost (experiment calls) |
| --- | --- | --- | --- | --- |
| code-rag | 1 | 48/48 labels; 6/6 top-1 queries | 335 ms | $0.00714 |
| code-rag | 4 | 48/48 labels; 6/6 top-1 queries | 494 ms | $0.00436 |
| customer-voice | 1 | 138/144 agreement with saved TypeSafe | 329 ms | $0.00904 |
| customer-voice | 4 | 139/144 agreement with saved TypeSafe | 498 ms | $0.01308 |

The successful remote function took 96.78 seconds, including 56.91 seconds of setup and a 9.23-second warmup. Its estimated GPU cost is $0.10617; this is not total billed project spend. The earlier failed-transfer run is excluded as documented above. The client remote-call wall time was 103.31 seconds. Peak VRAM was not measured. Batch size is one document, with six answer fields sharing the same canvas.

All five customer-support disagreements at four steps are against TypeSafe probabilities between 0.50 and 0.56, near the chosen decision threshold. This is not evidence that our judgments are wrong. Conversely, our restricted probabilities were extremely confident on these ambiguous decisions; agreement does not establish calibration.

The one-step retrieval timing includes slow outliers (p95 2.13 seconds); one global warmup does not eliminate all input-shape/setup effects. These are exploratory measurements, not a stable cost or speed comparison. See [BATCHING.md](BATCHING.md) for the subsequent batching and memory measurements.
