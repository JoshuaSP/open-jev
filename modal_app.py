"""CPU preflight: modal run modal_app.py
GPU smoke experiment: modal run modal_app.py --gpu
"""
import json
import os
from pathlib import Path

import modal

MODEL = "google/diffusiongemma-26B-A4B-it"
REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
app = modal.App("open-jev")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.0", "transformers==5.11.0", "accelerate", "huggingface_hub", "hf-xet", "Pillow")
    .pip_install("torchvision==0.29.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": "/cache/hub"})
    .add_local_python_source("constraints")
    .add_local_python_source("json_canvas")
)
cache = modal.Volume.from_name("open-jev-hf-cache", create_if_missing=True)
secret = modal.Secret.from_name("huggingface")


def hf_auth():
    # Normalize common secret key names without printing credentials.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    raise RuntimeError("The huggingface secret needs an HF_TOKEN-compatible key")


@app.function(image=image, secrets=[secret], volumes={"/cache": cache}, timeout=600)
def preflight():
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from constraints import ChoiceCanvasProcessor, label_tokens

    token = hf_auth()
    revision = REVISION
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, token=token)
    labels, ids = label_tokens(tokenizer, 3)
    # Adversarial logits overwhelmingly prefer a forbidden token.
    processor = ChoiceCanvasProcessor([ids[:2], ids], 4, len(tokenizer), tokenizer.eos_token_id, "cpu")
    logits = torch.randn(2, 4, len(tokenizer))
    logits[..., tokenizer.eos_token_id] = 10000
    masked = processor(None, logits)
    predictions = masked.argmax(-1).tolist()
    for row in predictions:
        assert row[0] in ids[:2] and row[1] in ids
        assert row[2:] == [tokenizer.eos_token_id] * 2
        processor.decode(row, [[False, True], ["low", "medium", "high"]])
    assert torch.isfinite(torch.distributions.Categorical(logits=masked).entropy()).all()
    try:
        processor.decode([tokenizer.eos_token_id, ids[0]], [[False, True], ["low", "medium", "high"]])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid decisions must fail closed")
    cache.commit()
    return {"model": MODEL, "revision": revision, "transformers": transformers.__version__,
            "labels": dict(zip(labels, ids)), "constraint_checks": "passed", "gpu_used": False}


@app.function(image=image, secrets=[secret], volumes={"/cache": cache}, timeout=600)
def inspect_json():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=hf_auth())
    rows = []
    for department in ("billing", "technical", "sales"):
        for refund in (False, True):
            text = json.dumps({"department": department, "refund_requested": refund})
            ids = tokenizer.encode(text, add_special_tokens=False)
            rows.append({"text": text, "ids": ids,
                         "pieces": [tokenizer.decode([i]) for i in ids]})
    from json_canvas import check_constraints
    return {"tokenization": rows, "tests": check_constraints(tokenizer, len(tokenizer), tokenizer.eos_token_id)}


@app.function(image=image, secrets=[secret], volumes={"/cache": cache}, gpu="H100", timeout=900)
def json_smoke():
    import time
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, LogitsProcessorList
    from json_canvas import JsonCanvas, FinalReadoutCanvas, constrained_sampler, routing_outputs, check_constraints

    token = hf_auth()
    revision = REVISION
    processor = AutoProcessor.from_pretrained(MODEL, revision=revision, token=token)
    tokenizer = processor.tokenizer
    checks = check_constraints(tokenizer, len(tokenizer), tokenizer.eos_token_id)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, revision=revision, token=token, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    cache.commit()
    instructions = (
        "Return only a JSON object with exactly these two fields:\n"
        '\"department\": one of \"billing\", \"technical\", \"sales\".\n'
        '\"refund_requested\": true if a refund is requested, otherwise false.\n'
        "Use the field order department, refund_requested. Do not include an explanation."
    )
    state = "I was charged twice for my subscription. Please refund the duplicate charge."
    prompts = {
        "original": "State: " + state + "\n" + instructions,
        "bracketed": "<user_text>\n" + state + "\n</user_text>\n\n<instructions>\n"
                     + instructions + "\n</instructions>",
    }
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    from contextlib import nullcontext

    class CountPasses:
        def __init__(self):
            self.steps = 0
        def __call__(self, input_ids, scores):
            self.steps += 1
            return scores

    def run(prompt_name, noise, seed, steps, readout="final"):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": prompts[prompt_name]}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        torch.manual_seed(seed)
        stock = noise == "stock"
        layout_class = FinalReadoutCanvas if readout == "final" else JsonCanvas
        layout = CountPasses() if stock else layout_class(
            tokenizer, routing_outputs(), model.config.canvas_length,
            model.config.text_config.vocab_size, eos, model.device)
        context = nullcontext() if stock else constrained_sampler(model, layout, noise=noise)
        with context:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.generate(
                **inputs, max_new_tokens=model.config.canvas_length, max_denoising_steps=steps,
                cache_implementation="dynamic", logits_processor=LogitsProcessorList([layout]),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        generated = result.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if stock:
            try:
                value = json.loads(text)
                valid = (isinstance(value, dict) and set(value) == {"department", "refund_requested"}
                         and value["department"] in ("billing", "technical", "sales")
                         and type(value["refund_requested"]) is bool)
            except (ValueError, TypeError):
                value, valid = None, False
        else:
            if readout == "final":
                generated = layout.select_final()[0].tolist()
            text, value = layout.decode(generated)
            valid = True
        return {"prompt": prompt_name, "noise": noise, "readout": "stock" if stock else readout, "seed": seed, "max_steps": steps,
                "actual_steps": layout.steps, "generation_seconds": elapsed, "text": text,
                "valid": valid,
                "correct": valid and value == {"department": "billing", "refund_requested": True}}

    warmup = run("bracketed", "vocabulary", 0, 1)
    records = [run("bracketed", "vocabulary", seed, steps)
               for seed in range(3) for steps in (1, 2, 4, 8)]
    # Preserve one intermediate-mask comparison and isolate prompt bracketing.
    records += [run("bracketed", "vocabulary", seed, 8, readout="each_step") for seed in range(3)]
    records += [run("original", "vocabulary", seed, 8) for seed in range(3)]
    records += [run("bracketed", "stock", seed, 48) for seed in range(3)]
    return {"model": MODEL, "revision": revision, "hardware": torch.cuda.get_device_name(),
            "prompts": prompts, "tests": checks, "warmup": warmup, "measurements": records,
            "note": "Single input, three seeds; constrained runs use exact step budgets; stock uses adaptive stopping."}


@app.function(image=image, secrets=[secret], volumes={"/cache": cache}, gpu="H100", timeout=900)
def smoke():
    import time
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, LogitsProcessorList
    from constraints import ChoiceCanvasProcessor, label_tokens

    token = hf_auth()
    revision = REVISION
    processor = AutoProcessor.from_pretrained(MODEL, revision=revision, token=token)
    tokenizer = processor.tokenizer
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, revision=revision, token=token, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    cache.commit()
    choices = [["billing", "technical", "sales"], [False, True]]
    labels, ids = label_tokens(tokenizer, 3)
    prompt = (
        "State: I was charged twice for my subscription. Please refund the duplicate charge.\n"
        "Return exactly two uppercase letters, with no spaces or explanation.\n"
        "First letter: department: A=billing, B=technical, C=sales.\n"
        "Second letter: refund requested: A=false, B=true."
    )
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    records = []
    # First call is warmup; these are smoke timings, not benchmark statistics.
    for steps in (1, 1, 2, 4, 8):
        torch.manual_seed(0)
        constraint = ChoiceCanvasProcessor(
            [ids, ids[:2]], model.config.canvas_length,
            model.config.text_config.vocab_size, eos, model.device,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.generate(
            **inputs, max_new_tokens=model.config.canvas_length,
            max_denoising_steps=steps, cache_implementation="dynamic",
            logits_processor=LogitsProcessorList([constraint]),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = result.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        values = constraint.decode(generated, choices)
        records.append({"max_steps": steps, "actual_steps": constraint.steps,
                        "generation_seconds": elapsed, "values": values,
                        "correct": values == ["billing", True],
                        "uncalibrated_choice_probabilities": [
                            x.softmax(-1)[0].cpu().tolist() for x in constraint.last_choice_logits
                        ]})
    return {"model": MODEL, "revision": revision, "hardware": torch.cuda.get_device_name(),
            "warmup": records[0], "measurements": records[1:],
            "note": "One example, eager BF16, fixed 256-token canvas; not a Jev benchmark."}


@app.local_entrypoint()
def main(gpu: bool = False, inspect: bool = False, json_mode: bool = False):
    if inspect:
        result, filename = inspect_json.remote(), "json_tokenization.json"
    elif gpu and json_mode:
        result, filename = json_smoke.remote(), "json_final_readout.json"
    elif gpu:
        result, filename = smoke.remote(), "smoke.json"
    elif json_mode:
        raise ValueError("Use --gpu --json-mode for the JSON inference experiment")
    else:
        result, filename = preflight.remote(), "preflight.json"
    Path("results").mkdir(exist_ok=True)
    path = Path("results") / filename
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.read_text())
