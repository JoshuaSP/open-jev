"""Pack public-eval questions into finite JSON canvases and batch across nodes."""

import json, time, collections
from datetime import datetime, timezone
from pathlib import Path
import modal
from modal_app import MODEL, REVISION, image, cache, secret, hf_auth

ROOT = Path(__file__).parent
RATE = 0.001097
app = modal.App("open-jev-parallel-evals")
image = image.add_local_python_source("modal_app", "parallel_canvas")


def load_nodes():
    nodes = []
    for path in sorted((ROOT / "typesafe-public-evals").glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text())
        for cid, case in data["cases"].items():
            for node in case["models"]["typesafe"]["nodes"]:
                if not node["ran"]:
                    continue
                fields = {}
                for key, idx in node["questions"].items():
                    q = data["questions"][idx]
                    options = (
                        [False, True]
                        if q["type"] == "noul"
                        else list(range(len(q["criteria"])))
                        if q["type"] == "score"
                        else list(q["criteria"])
                    )
                    fields[key] = dict(
                        question=q["instructions"],
                        criteria=q["criteria"],
                        options=options,
                        type=q["type"],
                    )
                nodes.append(
                    dict(
                        id="/".join((path.stem, cid, node["node"])),
                        workflow=path.stem,
                        case=cid,
                        document=json.dumps(
                            data["documents"][node["doc"]], ensure_ascii=False
                        ),
                        fields=fields,
                    )
                )
    return nodes


@app.function(
    image=image, secrets=[secret], volumes={"/cache": cache}, gpu="H100", timeout=1800
)
def evaluate(nodes, steps=1):
    started = time.perf_counter()
    import gc, torch, transformers
    from transformers import (
        AutoProcessor,
        DiffusionGemmaForBlockDiffusion,
        LogitsProcessorList,
    )
    from parallel_canvas import compile_fields, FieldCanvas, BatchCanvas
    from json_canvas import constrained_sampler

    token = hf_auth()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, token=token)
    processor.tokenizer.padding_side = "left"
    jobs = []

    def add(node, fields):
        instructions = "Answer all questions below. Return one JSON object with exactly the listed keys and allowed values. Score criteria use zero-based indices. No explanations.\n"
        for key, q in fields.items():
            instructions += (
                json.dumps(key)
                + ": "
                + q["question"]
                + "\nCriteria: "
                + json.dumps(q["criteria"], ensure_ascii=False)
                + "\nAllowed values: "
                + json.dumps(q["options"])
                + "\n"
            )
        prompt = (
            "<user_text>\n"
            + node["document"]
            + "\n</user_text>\n\n<instructions>\n"
            + instructions
            + "\n</instructions>"
        )
        encoded = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        jobs.append(
            dict(
                node_id=node["id"],
                workflow=node["workflow"],
                case=node["case"],
                fields=fields,
                prompt=prompt,
                input_tokens=int(encoded["attention_mask"].sum()),
            )
        )

    for node in nodes:
        fields = {}
        for key, q in node["fields"].items():
            trial = dict(fields, **{key: q})
            try:
                compile_fields(
                    processor.tokenizer, {k: v["options"] for k, v in trial.items()}
                )
            except ValueError:
                if not fields:
                    raise
                add(node, fields)
                fields = {key: q}
                compile_fields(processor.tokenizer, {key: q["options"]})
            else:
                fields = trial
        if fields:
            add(node, fields)
    print(
        f"Packed {sum(len(n['fields']) for n in nodes)} questions from {len(nodes)} nodes into {len(jobs)} canvases",
        flush=True,
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, revision=REVISION, token=token, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    setup = time.perf_counter() - started
    jobs.sort(key=lambda x: x["input_tokens"])
    outputs = []
    records = []
    failures = []
    offset = 0
    size = 64
    while offset < len(jobs):
        chunk = jobs[offset : offset + size]

        def run():
            start = time.perf_counter()
            layouts = [
                FieldCanvas(
                    processor.tokenizer,
                    {k: v["options"] for k, v in job["fields"].items()},
                    model.config.canvas_length,
                    model.config.text_config.vocab_size,
                    eos,
                    model.device,
                )
                for job in chunk
            ]
            batched = BatchCanvas(layouts)
            inputs = processor.apply_chat_template(
                [[{"role": "user", "content": j["prompt"]}] for j in chunk],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            ).to(model.device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), constrained_sampler(model, batched):
                model.generate(
                    **inputs,
                    max_new_tokens=model.config.canvas_length,
                    max_denoising_steps=steps,
                    cache_implementation="dynamic",
                    logits_processor=LogitsProcessorList([batched]),
                )
            decoded = [
                layout.decode(layout.select_final()[0].tolist()) for layout in layouts
            ]
            torch.cuda.synchronize()
            stat = dict(
                request_seconds=time.perf_counter() - start,
                batch_size=len(chunk),
                requested_batch_size=size,
                input_tokens=int(inputs["attention_mask"].sum()),
                padded_input_tokens=int(inputs["input_ids"].numel()),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            return decoded, stat

        try:
            decoded, stat = run()
        except torch.cuda.OutOfMemoryError as error:
            failures.append(dict(batch_size=len(chunk), target=size, error=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size //= 2
            print(f"OOM; reducing batch target to {size}", flush=True)
            continue
        records.append(stat)
        for job, (text, values) in zip(chunk, decoded):
            outputs.append(
                dict(
                    node_id=job["node_id"],
                    workflow=job["workflow"],
                    case=job["case"],
                    fields=job["fields"],
                    values=values,
                    json=text,
                    input_tokens=job["input_tokens"],
                )
            )
        offset += len(chunk)
        print(f"Completed {offset}/{len(jobs)} canvases", flush=True)
    return dict(
        outputs=outputs,
        records=records,
        failures=failures,
        setup_seconds=setup,
        remote_function_seconds=time.perf_counter() - started,
        model=MODEL,
        revision=REVISION,
        steps=steps,
        seed=0,
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        hardware=torch.cuda.get_device_name(),
    )


@app.local_entrypoint()
def main(steps: int = 1):
    if steps < 1:
        raise ValueError("Steps must be positive")
    from public_eval_benchmark import load_cases, summarize

    _, labels, sources, saved = load_cases()
    nodes = load_nodes()
    start = time.perf_counter()
    result = evaluate.remote(nodes, steps)
    result["client_seconds"] = time.perf_counter() - start
    predictions = []
    merged = {}
    for out in result["outputs"]:
        target = merged.setdefault(out["node_id"], {})
        for key, value in out["values"].items():
            assert key not in target
            target[key] = value
            predictions.append(
                dict(
                    id=out["node_id"] + "/" + key,
                    workflow=out["workflow"],
                    case=out["case"],
                    type=out["fields"][key]["type"],
                    value=value,
                    valid=True,
                )
            )
    assert {r["id"] for r in predictions} == set(labels)
    result.update(
        predictions=predictions,
        node_outputs=merged,
        source_sha256=sources,
        jev_saved_case_costs=saved,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )
    result["summary"] = summarize(result, labels)
    total = sum(r["request_seconds"] for r in result["records"])
    tokens = sum(r["input_tokens"] for r in result["records"])
    result["inference_totals"] = dict(
        seconds=total,
        input_tokens=tokens,
        gpu_usd=total * RATE,
        gpu_usd_per_million_input_tokens=total * RATE * 1e6 / tokens,
    )
    result["estimated_remote_function_gpu_usd"] = (
        result["remote_function_seconds"] * RATE
    )
    result["cost_scope"] = (
        "GPU-only estimate at $3.9492/hour, excludes CPU/RAM/storage and lifecycle outside function. Includes layout construction in request timing; first-request overhead not excluded."
    )
    result["scope"] = (
        "All original 408 questions; grouped within saved Jev nodes, split only at 256-token canvas limit. Shared attention across fields. Not independent workflow branching."
    )
    path = (
        ROOT
        / "results"
        / (
            "parallel_evals_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                path=str(path),
                summary=result["summary"],
                inference=result["inference_totals"],
                total_gpu_usd=result["estimated_remote_function_gpu_usd"],
            ),
            indent=2,
        )
    )
