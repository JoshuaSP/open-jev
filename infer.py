"""Modal CLI for the reusable harness: modal run infer.py --input examples/routing.json"""
import json
import time
from pathlib import Path
import modal
from modal_app import image, cache, secret, hf_auth, MODEL, REVISION
from inference import prepare

app = modal.App('open-jev-inference')
image = image.add_local_python_source('modal_app', 'inference')


@app.function(image=image, secrets=[secret], volumes={'/cache': cache}, gpu='H100', timeout=900)
def run(requests, steps, batch_size):
    import gc
    import torch
    from inference import DiffusionHarness
    started = time.perf_counter()
    harness = DiffusionHarness.load(token=hf_auth())
    setup = time.perf_counter()-started
    batches, failures = [], []
    offset = 0
    while offset < len(requests):
        try:
            result = harness.predict(requests[offset:offset+batch_size], steps=steps)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            failures.append(batch_size)
            batch_size //= 2
            gc.collect()
            torch.cuda.empty_cache()
            continue
        batches.append(result)
        offset += len(result['answers'])
    seconds = time.perf_counter()-started
    return dict(model=MODEL, revision=REVISION, batches=batches, oom_batch_sizes=failures,
                setup_seconds=setup, remote_seconds=seconds, estimated_gpu_usd=seconds*.001097,
                cost_scope='H100 GPU only at $3.9492/hour; excludes CPU/RAM/storage and lifecycle outside function')


@app.local_entrypoint()
def main(input: str = 'examples/routing.json', steps: int = 1, batch_size: int = 64, output: str = 'results/inference.json'):
    payload = json.loads(Path(input).read_text())
    requests = payload if isinstance(payload, list) else [payload]
    if not requests or steps < 1 or batch_size < 1:
        raise ValueError('Need requests, positive steps and positive batch size')
    for request in requests:
        prepare(request)
    if any(r['questions'] != requests[0]['questions'] for r in requests):
        raise ValueError('Requests in one invocation must share a question schema')
    result = run.remote(requests, steps, batch_size)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
