# Open Jev

An experimental inference harness for **typed JSON decisions with DiffusionGemma**. Denoise freely, then choose the most likely allowed tokens from the final logits. No training or fine-tuning.

The goal is to explore how close an open diffusion model can get to TypeSafe AI's Jev on structured judgments. This is an independent experiment, not a reproduction of Jev's architecture, training, calibrated probabilities, or published workflow scores.

## Results at a glance

Measured September 16, 2026 on one H100 80GB, BF16, using `google/diffusiongemma-26B-A4B-it`. Results below are from **one denoising step**.

### Every's TypeSafe lab

| Task | Data | Result |
| --- | --- | --- |
| Code retrieval | 8 documents × 6 questions | **48/48** relevance labels; **6/6** top-1 retrieval |
| Customer voice | 24 messages × 6 questions | **138/144 (95.8%)** agreement with saved TypeSafe decisions |

Customer voice is agreement with a saved model, not human-labeled accuracy. We tested two of the bundle's eleven experiments. The four-step run achieved 139/144 support agreement and the same retrieval result.

With six boolean fields on each canvas, batch 16 achieved **30.41 documents/s**, **182.44 judgments/s**, and an estimated **$0.0361 per 1,000 documents** in warm GPU time. Batches 64 and 32 OOMed. This throughput workload repeated the 32 unique documents into 384 evaluations; it is not 384 independent examples. Prompt KV caches were not reused.

[Every results and methodology](EVERY_LAB.md) · [Batching, memory and costs](BATCHING.md)

### Jev's public examples

We replayed the original questions on the saved Jev execution paths, then compared both models to the same model-derived reference answers.

| Workflow | Scored questions | Ours | Saved Jev |
| --- | ---: | ---: | ---: |
| Agent traces | 52 | 78.8% | 78.8% |
| Customer service | 92 | 91.3% | 89.1% |
| Invoice processing | 167 | 92.8% | 97.0% |
| Security incidents | 26 | 69.2% | 80.8% |
| **Total** | **337** | **88.4%** | **90.8%** |

All **408/408** emitted answers satisfied their finite output constraints. Accuracy excludes 54 questions without references and 17 tied references. These are **20 selected public cases**, not the full 711-case corpus. We did not reproduce independent branch selection or final workflow decisions, and did not call the Jev API.

This runner puts one question in each sequence and repeats its document. Successful inference took **248 seconds / $0.272 GPU cost** across **2.214M input tokens** (about **$0.123/MTok**); including loading and OOM attempts, **339 seconds / $0.372**. Jev's saved whole-workflow records total 8.87 seconds and $0.00846 estimated API cost, but use different question grouping and execution. We have **not matched Jev's speed or cost** on these workflows.

[Public-eval results and limitations](PUBLIC_EVALS.md) · [Individual predictions](results/public_evals_20260916T193219Z.json)

All dollar figures are estimates at **$3.9492/H100-hour**, excluding CPU, RAM, storage and lifecycle outside the measured function. They are not invoice totals or hosted API prices.

## Quick start: Modal

Requires Python 3.11+, a Modal account, and a Hugging Face token with access to the model. Accept the model's access terms on Hugging Face first.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
modal setup
# If not already configured, create a Modal secret named "huggingface"
# containing HF_TOKEN. This opens your configured editor for the value:
modal secret create huggingface HF_TOKEN=-

# CPU-only tokenizer/constraint verification; does not load model weights
modal run verify.py

# Runs a bounded H100 job, not a persistent endpoint
modal run infer.py --input examples/routing.json --steps 1
```

The inference command writes `results/inference.json`, including answers, token counts, timing, peak memory and a GPU cost estimate. Model downloads persist in Modal volume `open-jev-hf-cache`; `HF_XET_HIGH_PERFORMANCE=1` enables fast Xet downloads. The first GPU run downloads weights and incurs loading time.

An input file contains one request, or an array of requests sharing the same question schema:

```json
{
  "document": "I was charged twice. Please refund the duplicate payment.",
  "questions": {
    "department": {
      "question": "Which department should handle this?",
      "options": ["billing", "technical", "sales"]
    },
    "refund_requested": {
      "question": "Does the customer request a refund?",
      "options": [false, true]
    }
  }
}
```

The output answer is actual JSON, such as `{"department": "billing", "refund_requested": true}`. There is no letter-code translation. Strings, integers and booleans are supported. The harness batches up to 64 available requests and halves on CUDA OOM; it does not duplicate requests to fill a batch.

For a local CUDA environment with enough VRAM:

```bash
pip install -e '.[inference]'
```

```python
import json
from inference import DiffusionHarness

request = json.load(open("examples/routing.json"))
harness = DiffusionHarness.load()  # uses your HF authentication
result = harness.predict([request], steps=1, seed=0)
print(result["answers"])
```

## How it works

1. Tokenize complete allowed JSON documents. Verify round trips and the finite candidate language.
2. Fix token positions shared by every candidate. Initialize other positions with uniform full-vocabulary noise, rather than random valid answers.
3. Run exactly N denoising steps on the model's 256-token canvas. Variable positions remain unrestricted during denoising and self-conditioning.
4. At the final readout, select the highest-logit allowed token. For multi-token alternatives, the first differing token discriminates; later branches resolve surviving candidates, and unique suffixes are forced.
5. Decode and validate the complete selected JSON document. No JSON repair or extra model pass.

The prompt brackets the document in `<user_text>` and places instructions afterward in `<instructions>`. These are formatting delimiters, not a security boundary.

`FinalReadoutCanvas` handles aligned independent slots. `FinalTrieCanvas` extends final readout to dependent and unequal-length alternatives. Later trie decisions use logits from the same diffusion pass, not autoregressive likelihoods conditioned on the newly selected prefix. With unequal token lengths, only structure common to every candidate can be fixed during denoising; final output is still constrained to a complete candidate.

The reusable harness enumerates up to 4,096 complete alternatives and requires the output to fit one canvas. It is not an arbitrary JSON Schema engine. Multiple fields share attention, so this does not reproduce Jev's independent-question isolation. Restricted softmax scores used in the Every ranking experiment are **uncalibrated**, not reliable truth probabilities.

## Reproduce the experiments

Run from the repository root. Historical receipts are included; new runs produce timestamped files under `results/`.

```bash
# Every: import your downloaded lab ZIP (source bundle is not redistributed)
python scripts/import_every.py /path/to/typesafe-lab.zip
modal run lab_benchmark.py       # 1- and 4-step runs, 32 documents per budget
modal run batch_benchmark.py     # 1-step, start 64; halve only on OOM

# Jev: downloaded public examples are already included with provenance
modal run public_eval_benchmark.py

# Local checks: no credentials or GPU needed
python -m unittest discover -s tests -v
```

The model revision is pinned to `f7f5b7f5fa82ffc52addd066915886d497f5517b`; the Modal image pins Transformers 5.11.0, Torch 2.14.0 and torchvision 0.29.0. GPU type, padding, batch composition and kernels can affect numerical results; bitwise reproduction is not promised.

The original public-eval run had a token-length bookkeeping bug: it counted the batch dimension, which also prevented length sorting. Exact token counts were recomputed on CPU without rerunning inference. The included receipt is corrected and preserves actual batch membership. The current runner fixes counting and sorting; timing and some numerical outputs may differ on rerun. `public_eval_token_audit.py` and `public_eval_report.py` retain the historical audit procedure, not a required rerun step.

## Repository map

| File | Purpose |
| --- | --- |
| `inference.py`, `infer.py` | Reusable harness and Modal CLI |
| `json_canvas.py` | Token constraints, final readout and sampler integration |
| `modal_app.py` | Pinned image/model, HF cache, historical smoke experiments |
| `lab_benchmark.py`, `batch_benchmark.py` | Every accuracy and throughput runners |
| `public_eval_benchmark.py` | Public Jev question replay and scoring |
| `results/` | Selected recorded experiment receipts |
| `typesafe-public-evals/` | Public examples, source URLs and checksums |
| `docs/RESEARCH_LOG.md` | Historical experiments, including abandoned approaches |

## Attribution and licensing

Project code is MIT licensed. Model weights remain subject to Google's terms. Third-party benchmark data and quoted questions retain their original rights; the MIT license does not cover those materials. See [THIRD_PARTY.md](THIRD_PARTY.md).

Sources: [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it), [Transformers sampler](https://github.com/huggingface/transformers/blob/v5.11.0/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py), [TypeSafe public evaluations](https://evals.typesafe.ai/), [Jev introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
