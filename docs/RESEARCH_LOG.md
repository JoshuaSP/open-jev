# Historical research log

This is an archived chronology. Earlier approaches were abandoned, and some original workspace artifacts referenced below are not bundled. Start with the root README for the current method and commands.

# Open Jev experiments

Hypothesis: an open diffusion language model can make several constrained,
typed decisions in parallel, trading accuracy for latency. This is an inference
experiment, not a reproduction of TypeSafe's undisclosed model or training.

## First experiment

We assign each question a canvas position and map its alternatives to verified
single-token labels (`A`, `B`, ...). At every denoising step we mask all other
tokens at that position. Python maps the chosen labels to the original values;
it never asks the model to spell JSON keys or arbitrary enum strings.

```sh
# Uses your installed Modal CLI and the existing `huggingface` secret.
modal run modal_app.py           # CPU: access, tokenizer, adversarial mask checks
modal run modal_app.py --gpu     # One H100, bounded 15-minute smoke experiment
```

Results are written under `results/`. Model files live in a persistent Modal
volume (`open-jev-hf-cache`). Downloads use `HF_XET_HIGH_PERFORMANCE=1`.
Only the GPU command downloads model weights. No service is deployed.

The GPU experiment has two decisions, one warmup, then 1/2/4/8-step runs.
It reports generation wall time (including prefill, excluding loading/network),
actual denoising steps, typed values, and uncalibrated restricted-label softmax.
It is one smoke example, not an accuracy or latency benchmark.

## What we know

DiffusionGemma encodes the prompt into a KV cache, then refines a 256-token
canvas using bidirectional attention. Finished canvases are encoded into the
context before the next canvas. The published recipe allows up to 48 denoising
steps with adaptive stopping. Token throughput for long text does not establish
latency for two decisions. [Google model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)

The Hugging Face v5.11.0 sampler initializes from **uniform random vocabulary
tokens**, rather than a repeated mask token. It predicts logits at every position,
applies custom logits processors, selects tokens using entropy-bounded sampling,
and fully renoises unselected positions. Processed logits also supply
self-conditioning. Final output uses the latest argmax canvas. Our constraints
apply to predictions; intermediate noise can still contain any vocabulary token.
[Sampler source](https://github.com/huggingface/transformers/blob/v5.11.0/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py)

Jev exposes Choice, Score, and Noul (truth probability); its docs promise
independent evaluation of questions against shared state. Our initial shared
canvas permits interaction between questions and therefore does not reproduce
that isolation. A separate experiment should batch one question per sequence,
then investigate sharing the state prefill.
[TypeSafe documentation](https://docs.typesafe.ai/introduction)

TypeSafe reports 70–500 ms end-to-end responses and explicitly attributes Jev
to a new architecture, sampler, and RLCD training. Its headline speedup is a
workflow comparison, not an established speedup over constrained DiffusionGemma.
Their workflow reference labels come from model consensus, not necessarily
human ground truth. They also provide an LLM adapter that can generate whole
probability distributions; that is different work from emitting just decisions.
[Launch and methodology](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## Guarantees and open questions

- For supported categorical slots, every successfully decoded value belongs
  to the caller's supplied alternatives. Invalid/incomplete outputs raise an
  error. This does not imply correct judgments, uptime, or arbitrary JSON Schema
  support. Booleans and finite ordinal scores fit this representation naturally.
- Independent per-position masks cannot generally enforce dependencies between
  fields or multi-token alternatives. For example, per-position unions for
  `AB` and `CD` also admit `AD`. Single-token labels avoid that particular issue.
- Restricted softmax is **not calibrated confidence**. Measure NLL, Brier score,
  and calibration on held-out labels, including ambiguous inputs and alternative
  label permutations. Temperature scaling needs a separate calibration split.
- One diffusion step may be fast but useless. Compare quality across step budgets
  before optimizing kernels or changing the trained canvas length.
- Masking 254 of 256 slots to EOS dilutes canvas-average entropy. Early stopping
  may trigger while the decision slots remain uncertain. Log actual steps and
  later replace this with decision-slot stopping criteria.
- The prototype retains the full vocabulary projection and the full canvas.
  Most compute remains. A restricted LM head and smaller canvases are experiments,
  not presumed drop-in optimizations, especially with self-conditioning.

## Evaluation sequence

1. Verify constrained model execution and inspect step/accuracy behavior.
2. Compare stock generation, constrained diffusion, and an autoregressive
   single-token classifier on identical states and questions.
3. Sweep 1/8/32/128 questions and short/long states. Report warm p50/p95 request
   latency, cold starts separately, decisions/sec, GPU time, and quality. Compare
   batched isolated questions against shared-canvas questions.
4. Add finite rubric scores and binary probabilities. Evaluate calibration
   separately from type validity and accuracy.
5. Reuse [TypeSafe's public workflows](https://evals.typesafe.ai/) and
   [adapter](https://github.com/typesafe-ai/system-one-adapter-python) for a
   closer comparison, reporting differences in hardware, pricing, and harness.

The first milestone is useful decisions at low step counts. Matching Jev's
latency and calibrated probabilities remains an empirical question.

## First measured result — 2026-09-15

CPU preflight passed: HF access, distinct label IDs, finite masked entropy,
and rejection of forbidden decoded values. Labels A/B/C map to
236776/236799/236780. Model revision:
`f7f5b7f5fa82ffc52addd066915886d497f5517b`.

The H100 80GB BF16 smoke run completed. The input explicitly requested a refund
for a duplicate subscription charge; expected decisions were `billing, true`.

| Step budget | Actual steps | Warm generation | Department | Refund requested |
| --- | --- | --- | --- | --- |
| 1 | 1 | 133 ms | technical | false |
| 2 | 2 | 195 ms | technical | true |
| 4 | 3 | 251 ms | technical | true |
| 8 | 3 | 252 ms | technical | true |

All returned values satisfied the constraints; none of the four runs got both
decisions right. The first warmup generation took 6.98 seconds (not included
above). These are single observations, not latency percentiles, and exclude
network, model loading, tokenization, and final response serialization.
The run used Transformers 5.11.0, Torch 2.14.0, and torchvision 0.29.0.
Raw data: `results/smoke.json`.

Interpretation: the latency is promising, but this naive forced-label canvas is
not yet useful. In the 4-step run the wrong department had restricted softmax
above 99.98%, directly illustrating why it must not be described as calibrated
confidence. Next compare natural answer templates and isolated questions,
disable canvas-average stopping, and retain an unconstrained model baseline.
Also verify full-template tokenization: individually valid label tokens need
not match how a tokenizer segments their concatenated text.

## Direct JSON canvas

Run `modal run modal_app.py --gpu --json-mode` for the direct JSON experiment,
or `modal run modal_app.py --inspect` for tokenization and CPU constraint checks.
The implementation is in `json_canvas.py`; results go to `results/json_smoke.json`.
The exact prompt and rendered chat template are included in the result file.

All six allowed JSON documents have 13 tokens, with exactly two variable slots:

```text
0    {"
1    department
2    ":
3     "
4    billing | technical | sales
5    ",
6     "
7    refund
8    _
9    requested
10   ":
11    false |  true
12   }
13+  EOS
```

The spaces shown at positions 3, 6, and 11 belong to their tokens. Department
IDs are billing=103547, technical=79205, sales=26377. Boolean IDs are
` false`=2416 and ` true`=1847. These were verified by tokenizing the complete
JSON documents, not by assuming standalone word tokenization carries over.

The structural tokens are fixed in the initial canvas and every re-noising
operation. Only value slots receive random allowed-value tokens. Logits are
masked to the same allowed sets on each pass. The final text is decoded and
parsed with `json.loads`; no label-to-value translation or JSON repair is used.
Early stopping is disabled so every requested denoising step actually executes.
This differs from the original letter experiment in prompt, canvas initialization,
constraints, and stopping, so any quality change is not an isolated ablation.

The small finite-language compiler verifies that independent token masks describe
exactly the allowed documents before using its parallel path. For dependent or
unequal-length multi-token alternatives, it falls back to a greedy candidate-trie
projection: choose the first discriminating token from current-pass logits, force
unique continuations, and resolve later branches among surviving candidates.
This supports complete alternatives without mixing suffixes. It is not a full
JSON Schema compiler, and enumerating every document will not scale to many fields.
The fallback's logits at later positions come from the same diffusion pass; they
are not autoregressive log probabilities conditioned on the newly selected prefix.
A later pass can inspect how suffix predictions change with the selected prefix.

### Direct JSON measurements — 2026-09-16

H100 80GB, BF16, same model revision as the original experiment. All 12 warm
outputs were allowed JSON documents and parsed without repairs. Three seeds on
one input are a sensitivity check, not an accuracy benchmark.

| Exact steps | Median warm generation (3 seeds) | Both fields correct |
| --- | --- | --- |
| 1 | 107 ms | 0/3 |
| 2 | 159 ms | 1/3 |
| 4 | 263 ms | 1/3 |
| 8 | 472 ms | 1/3 |

Seed 0 returned the correct billing/true object at 2, 4, and 8 steps; seeds 1
and 2 still returned false for refund_requested at 8 steps. The first warmup
call took 7.12 seconds. Generation timings include prefill but exclude model
loading, prompt processing, network, and final parsing. The constraint checks
covered all six exact outputs, finite masked entropy, valid initialization,
correlated alternatives, and multi-token alternatives of different lengths.
The observed seed sensitivity warrants investigating initialization and sampler
behavior before drawing conclusions about the underlying model's task accuracy.

## Current method: choose only at final readout

The previous valid-answer initialization and intermediate value masking are now
controls, not the default JSON experiment. `modal run modal_app.py --gpu --json-mode`
writes `results/json_final_readout.json` and uses `FinalReadoutCanvas`:

1. Fix the JSON keys, punctuation, and tail EOS tokens.
2. Initialize answer slots from uniform full-vocabulary noise, matching the stock
   sampler's distribution; re-noising uses that same distribution.
3. Run exactly N denoising steps. Value-slot logits are not masked: sampling and
   self-conditioning can use the entire vocabulary. Only structural logits are fixed.
4. Read the final raw logits at the two answer positions. Choose the argmax among
   billing/technical/sales and the argmax among false/true. No additional model pass,
   stochastic final choice, or confidence threshold is used.
5. Decode the resulting JSON token sequence and validate it.

This readout currently requires aligned independent single-token choices; the older
candidate-tree fallback remains available in `JsonCanvas` for intermediate masking.
The input is enclosed in `<user_text>...</user_text>`, followed by a separate
`<instructions>...</instructions>` block. Both remain in one user message.
Controls include eight-step intermediate masking with full-vocabulary noise,
final-only selection with the original prompt, and unmodified stock value/structure
sampling on the bracketed prompt (48-step ceiling with stock adaptive stopping).
All use the same model revision within the run. Tests verify that final-only
processing leaves value logits unchanged, even when a forbidden token dominates,
and that the final readout selects the correct allowed argmax for all six documents.

### Final-readout results — 2026-09-16

For the same billing/refund example, bracketed input and final-only selection
returned billing/true for all 12 runs (three seeds per budget):

| Exact steps | Median generation time | Both fields correct |
| --- | --- | --- |
| 1 | 134 ms | 3/3 |
| 2 | 198 ms | 3/3 |
| 4 | 330 ms | 3/3 |
| 8 | 591 ms | 3/3 |

Timing excludes final allowed-value readout, parsing, prompt tokenization, network,
model loading, and the initial warmup. These observations on one input are not
accuracy estimates or production latency percentiles.

At eight steps, intermediate masking with vocabulary noise also got 3/3, as did
final-only readout using the original unbracketed prompt. These controls do not
establish which change caused the improvement at one step. The stock sampler
expressed the correct billing/true values in all three outputs, but included
`thought` and Markdown fences. Consequently, its raw results have `valid=false`
and `correct=false`: those flags require an exact bare JSON response, and should
not be interpreted as wrong underlying decisions. Stock runs stopped after
4/2/3 passes respectively.

The readout performs two independent restricted argmax operations on final raw
logits. It does not sample an allowed answer, use an additional model pass, or
feed the selected answer back into denoising. More varied inputs and counterexamples
are necessary before making accuracy claims.
