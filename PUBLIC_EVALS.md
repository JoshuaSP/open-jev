# Public TypeSafe examples: one-step DiffusionGemma vs saved Jev

Run date: 2026-09-16. This is a question-level replay of the nodes Jev ran on 20 selected public examples, not a reproduction of independent workflow branching or their full 711-case evaluation.

## Accuracy

| Workflow | Scored questions | Ours | Jev |
| --- | ---: | ---: | ---: |
| agent_trace_observability | 52 | 41/52 (78.8%) | 41/52 (78.8%) |
| ALL | 337 | 298/337 (88.4%) | 306/337 (90.8%) |
| customer_service | 92 | 84/92 (91.3%) | 82/92 (89.1%) |
| invoice_processing | 167 | 155/167 (92.8%) | 162/167 (97.0%) |
| security_incidents | 26 | 18/26 (69.2%) | 21/26 (80.8%) |

All 408 answers were evaluated for output validity; 408 were valid. We exclude 54 questions without references and 17 tied reference answers from accuracy. References are model-derived, not human ground truth.

| Answer type | Scored | Ours correct | Jev correct |
| --- | ---: | ---: | ---: |
| noul | 220 | 199 | 205 |
| score | 27 | 19 | 20 |
| choice | 90 | 80 | 81 |

## Runtime and cost

- Successful inference requests: **248.03 seconds**, **$0.2721 GPU cost**. Includes first-request overhead; no separate warmup is excluded.
- GPU function including setup, OOM attempts and cleanup: **339.07 seconds**, **$0.3720**.
- Client wall time: **348.32 seconds**.
- Actual prompt tokens: **2,214,128** including templates, instructions and repeated documents. Successful inference cost: **$0.1229/million input tokens**.
- H100 rate: $0.001097/second ($3.9492/hour), GPU-only estimate. Excludes CPU/RAM/storage, the CPU token audit, and lifecycle outside the function. Not an invoice total.
- Batch target started at 64; caught OOM targets: [64, 32, 16, 8]. Actual successful batch sizes: [1, 2, 3, 4, 8].
- Peak allocated CUDA memory: 72.26 GiB; peak reserved: 77.88 GiB.

For context, Jev's saved whole-workflow records sum to 8.87 seconds and $0.008462 estimated API cost. This is not an equivalent speed/cost measurement: our runner repeats the document once per question, while their workflow groups questions. We did not call the Jev API.

## Method

- Original public documents, question wording and criteria. Each request asks for a literal JSON object containing `answer`, with the original boolean, score or named choice; no letter-label indirection.
- One denoising step, seed 0 full-vocabulary noise, fixed shared structural tokens. Final logits select the first differing allowed token; later branching positions discriminate remaining candidates and unique suffixes are forced. This is greedy candidate selection, not sequence likelihood scoring.
- Saved Jev model: `typesafe:v13_snowy_elephant`. Boolean labels use its probability argmax; choices and scores use the mode of the saved distribution.
- Reference distributions are averaged exactly as in the public viewer; missing distributions become one-hot votes. Reference ties are excluded. Numeric scores are evaluated as category argmax, not expected-value error.
- The inference worker receives no saved answers or references. Node selection and follow-up document context come from the saved Jev path, so this does not test whether our model would open the right branches.
- Public cases were selected to illustrate disagreements and agreements. These percentages do not estimate performance on the undisclosed corpus.
- Token counts were audited and corrected on CPU. The first GPU run mistakenly used the batch dimension for length sorting; predictions and timings are unchanged. The runner now counts attention-mask tokens correctly. Original batch membership is preserved in the receipt.

## Artifacts

- [Full receipt and individual predictions](results/public_evals_20260916T193219Z.json)
- [Runner](public_eval_benchmark.py)
- [Token audit](public_eval_token_audit.py)
- [Source data provenance](typesafe-public-evals/README.md)
- [TypeSafe public evaluations](https://evals.typesafe.ai/)
