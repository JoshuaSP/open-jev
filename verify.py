"""CPU verification using the real tokenizer and adversarial logits; no weights."""
import modal
from modal_app import image, cache, secret, hf_auth, MODEL, REVISION

app = modal.App('open-jev-verify')
image = image.add_local_python_source('modal_app', 'inference')


@app.function(image=image, secrets=[secret], volumes={'/cache':cache}, timeout=300)
def check():
    import json
    import torch
    from transformers import AutoTokenizer
    from json_canvas import FinalTrieCanvas, FinalReadoutCanvas, check_constraints
    from inference import prepare
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=hf_auth())
    result = check_constraints(tokenizer, len(tokenizer), tokenizer.eos_token_id)
    # Multi-token, shared-prefix, unequal-length choices must select a whole
    # candidate while forbidden logits dominate unrestricted denoising.
    options = ['billing support', 'billing escalation team', 'sales']
    texts = [json.dumps({'answer': x}) for x in options]
    layout = FinalTrieCanvas(tokenizer, texts, 64, len(tokenizer), tokenizer.eos_token_id, 'cpu')
    scores = torch.zeros(3, 64, len(tokenizer))
    scores[..., tokenizer.eos_token_id] = 10000
    for i, candidate in enumerate(layout.sequences):
        for p in layout.variable_positions:
            scores[i,p,candidate[p]] = 20000
    processed = layout(None,scores)
    assert torch.equal(processed[:,layout.variable_positions], scores[:,layout.variable_positions])
    assert [layout.decode(ids)[0] for ids in layout.select_final().tolist()] == texts
    # Trie and independent-slot argmax agree for a Cartesian boolean language.
    _, texts = prepare({'document':'test','questions':{'x':{'question':'x','options':[False,True]},'y':{'question':'y','options':[False,True]}}})
    args = (tokenizer,texts,32,len(tokenizer),tokenizer.eos_token_id,'cpu')
    trie, independent = FinalTrieCanvas(*args), FinalReadoutCanvas(*args)
    scores = torch.randn(2,32,len(tokenizer))
    trie(None,scores); independent(None,scores)
    assert torch.equal(trie.select_final(), independent.select_final())
    return dict(base_checks=result, final_trie='passed', gpu_used=False, revision=REVISION)


@app.local_entrypoint()
def main():
    print(check.remote())
