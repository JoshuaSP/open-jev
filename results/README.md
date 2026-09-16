# Recorded results

These receipts are the original measured runs, not results from the packaging/refactor work. GPU benchmarks were not rerun merely to assemble this repository.

- `every_lab_20260916T145029Z.json`: 32 unique documents at one and four steps, including prompts, source hashes, relevance labels and saved TypeSafe scores.
- `batch_sweep_20260916T145918Z.json`: one-step Every batching. 384 repeated document evaluations; 32 unique documents. Batch 64 and 32 OOM; 16 fits. Includes exclusions for earlier attempts.
- `public_evals_20260916T193219Z.json`: 408 questions on 20 selected public cases; 337 scored. Token accounting corrected by CPU audit; predictions and timings unchanged. Actual batch membership is recorded.
- `public_eval_token_audit.json`: exact input-token counts for the historical public-eval prompts.

Dollar amounts are GPU-rate estimates, not billed totals. See the linked reports at the repository root for timing boundaries, exclusions and interpretation. New result files are ignored by Git by default so that credentials or private prompts from future use are not inadvertently committed.
