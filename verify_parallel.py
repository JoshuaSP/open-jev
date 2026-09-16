import modal
from modal_app import image, cache, secret, hf_auth, MODEL, REVISION

app = modal.App("open-jev-verify-parallel")
image = image.add_local_python_source("modal_app", "parallel_canvas")


@app.function(image=image, secrets=[secret], volumes={"/cache": cache}, timeout=300)
def check():
    import torch
    from transformers import AutoTokenizer
    from parallel_canvas import FieldCanvas, BatchCanvas

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=hf_auth())
    fields = {
        "department": ["billing support", "billing escalation team", "sales"],
        "refund": [False, True],
        "severity": [0, 1, 2, 3],
    }
    layouts = [
        FieldCanvas(tok, fields, 128, len(tok), tok.eos_token_id, "cpu")
        for _ in range(2)
    ]
    logits = torch.zeros(2, 128, len(tok))
    logits[:, :, tok.eos_token_id] = 10000
    for row, layout in enumerate(layouts):
        for offset, candidates, variables in layout.slots:
            candidate = candidates[-1 if row else 0]
            for p, j, allowed in variables:
                logits[row, p, candidate[j]] = 20000
    batch = BatchCanvas(layouts)
    noisy = batch.initialize_canvas(2, "cpu")
    processed = batch(None, logits)
    for row, layout in enumerate(layouts):
        assert torch.equal(
            processed[row, layout.variable_positions],
            logits[row, layout.variable_positions],
        )
        value = layout.decode(layout.select_final()[0].tolist())[1]
        assert value == {k: v[-1 if row else 0] for k, v in fields.items()}
    # 30 booleans imply over a billion combinations, compiled without enumeration.
    large = FieldCanvas(
        tok,
        {f"x{i}": [False, True] for i in range(30)},
        256,
        len(tok),
        tok.eos_token_id,
        "cpu",
    )
    assert len(large.slots) == 30
    return {"checks": "passed", "gpu_used": False, "large_schema_fields": 30}


@app.local_entrypoint()
def main():
    print(check.remote())
