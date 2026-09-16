# TypeSafe public evaluation examples

Downloaded from https://evals.typesafe.ai/ on 2026-09-16. See `manifest.json` for exact source URLs and checksums. Each JSON file preserves the evaluation object from its public viewer data payload, including documents, typed questions, policies, reference answers, and saved model results.

| Workflow | Reported full evaluation cases | Public cases downloaded |
| --- | ---: | ---: |
| Agent trace observability | 117 | 5 |
| Security incidents | 240 | 5 |
| Invoice processing | 150 | 5 |
| Customer service | 204 | 5 |
| Total | 711 | 20 |

These are selected diagnostic examples, not a representative sample: each workflow shows one example where each of three models differs, one where all miss the reference, and one where all agree. Scores on these examples must not be presented as reproductions of the full published benchmark.

The inputs and typed questions can be used to test our constrained inference. Comparing final workflow decisions also requires reproducing the conditional question rounds and policy computations. The viewer data is not itself an executable benchmark harness. Reference answers are model-derived, not independent human ground truth.

This directory contains downloaded public data, not a license grant from TypeSafe. A one-step GPU replay of Jev's question nodes is recorded in [PUBLIC_EVALS.md](../PUBLIC_EVALS.md): 408 valid answers, 337 scored questions, 88.4% reference agreement for our model versus 90.8% for saved Jev answers. This does not reproduce final policy decisions.
