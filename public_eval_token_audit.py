"""CPU-only token audit of the exact public-eval prompts; no model weights."""
import json
from pathlib import Path
import modal
from modal_app import MODEL, cache, hf_auth, image, secret

app = modal.App('open-jev-public-token-audit')
image = image.add_local_python_source('modal_app')


@app.function(image=image, secrets=[secret], volumes={'/cache': cache}, timeout=300)
def count(rows):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, revision='f7f5b7f5fa82ffc52addd066915886d497f5517b', token=hf_auth())
    counts = {}
    old_counts = {}
    for r in rows:
        encoded = processor.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        counts[r['id']] = int(encoded['attention_mask'].sum())
        old_counts[r['id']] = len(processor.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                    tokenize=True, add_generation_prompt=True))
    return dict(counts=counts, original_count_method=old_counts)


@app.local_entrypoint()
def main():
    from public_eval_benchmark import load_cases
    rows, _, _, _ = load_cases()
    result = count.remote(rows)
    path = Path(__file__).parent / 'results/public_eval_token_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('Total:', sum(result['counts'].values()), 'Original:', sum(result['original_count_method'].values()))
