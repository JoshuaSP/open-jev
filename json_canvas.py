"""Finite JSON language constraints. Enumeration is for small experiments only."""
import itertools
import json
import math
from contextlib import contextmanager

import torch


def routing_outputs():
    return [json.dumps({"department": department, "refund_requested": refund})
            for department, refund in itertools.product(
                ("billing", "technical", "sales"), (False, True))]


class JsonCanvas:
    def __init__(self, tokenizer, texts, canvas_length, vocab_size, eos, device):
        if not texts or len(set(texts)) != len(texts):
            raise ValueError("Need distinct complete JSON alternatives")
        sequences = []
        for text in texts:
            json.loads(text)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= canvas_length or eos in ids:
                raise ValueError("JSON must fit on one canvas with room for EOS")
            if tokenizer.decode(ids) != text:
                raise ValueError("JSON must round-trip through the tokenizer")
            sequences.append(ids + [eos] * (canvas_length - len(ids)))
        self.tokenizer = tokenizer
        self.texts = set(texts)
        self.eos = eos
        self.sequences = torch.tensor(sequences, device=device)
        self.columns = [sorted(set(row[i] for row in sequences)) for i in range(canvas_length)]
        # Position-wise masks are exact ONLY if the language is a Cartesian product.
        self.factorized = math.prod(map(len, self.columns)) == len(set(map(tuple, sequences)))
        self.variable_positions = [i for i, ids in enumerate(self.columns) if len(ids) > 1]
        self.mask = torch.zeros((canvas_length, vocab_size), dtype=torch.bool, device=device)
        for i, ids in enumerate(self.columns):
            self.mask[i, ids] = True
        self.steps = 0
        self.last_value_logits = None

    def __call__(self, input_ids, scores):
        if scores.ndim != 3 or scores.shape[1:] != self.mask.shape:
            raise ValueError("Expected whole-canvas diffusion logits")
        self.steps += 1
        self.last_value_logits = [scores[:, i, self.columns[i]].detach().float().clone()
                                  for i in self.variable_positions]
        if self.factorized:
            return scores.masked_fill(~self.mask[None], -torch.inf)
        # Small-language fallback: walk the candidate trie using this pass's
        # logits. First differing token discriminates; unique suffixes are forced.
        # This is greedy projection, NOT an autoregressive sequence likelihood.
        active = torch.ones((scores.shape[0], len(self.sequences)), dtype=torch.bool, device=scores.device)
        for position in self.variable_positions:
            tokens = self.sequences[:, position]
            candidate_scores = scores[:, position, tokens].masked_fill(~active, -torch.inf)
            chosen = tokens[candidate_scores.argmax(-1)]
            active = active & (tokens[None] == chosen[:, None])
        selected = self.sequences[active.long().argmax(-1)]
        return torch.full_like(scores, -torch.inf).scatter(
            -1, selected[..., None], scores.gather(-1, selected[..., None]))

    def initialize_valid_canvas(self, batch_size, device):
        if not self.factorized:
            # Complete candidate initialization avoids hybrids with dependent slots.
            return self.sequences[torch.randint(len(self.sequences), (batch_size,), device=device)].clone()
        canvas = self.sequences[0].expand(batch_size, -1).clone()
        for position in self.variable_positions:
            allowed = torch.tensor(self.columns[position], device=device)
            canvas[:, position] = allowed[torch.randint(len(allowed), (batch_size,), device=device)]
        return canvas

    def initialize_canvas(self, batch_size, device):
        # Match the stock sampler's uniform-vocabulary noise in variable slots.
        # Structure remains fixed; intermediate noise is not a valid answer.
        canvas = torch.randint(self.mask.shape[1], (batch_size, self.mask.shape[0]), device=device)
        fixed = torch.tensor([len(ids) == 1 for ids in self.columns], device=device)
        return torch.where(fixed[None], self.sequences[0][None], canvas)

    def decode(self, ids):
        if self.eos not in ids:
            raise ValueError("Missing EOS")
        text = self.tokenizer.decode(ids[:ids.index(self.eos)])
        if text not in self.texts:
            raise ValueError("Output is not one of the complete allowed JSON documents")
        return text, json.loads(text)


@contextmanager
def constrained_sampler(model, layout, noise="vocabulary"):
    """Keep the template fixed in noise; use exact step budgets for this experiment."""
    if noise not in ("vocabulary", "valid"):
        raise ValueError("Unknown noise distribution")
    original_sampler = model._prepare_sampler
    original_stopping = model._prepare_diffusion_stopping_criteria

    def prepare(config):
        sampler = original_sampler(config)
        sampler.initialize_canvas = (layout.initialize_canvas if noise == "vocabulary"
                                     else layout.initialize_valid_canvas)
        return sampler

    model._prepare_sampler = prepare
    model._prepare_diffusion_stopping_criteria = lambda config: None
    try:
        yield
    finally:
        model._prepare_sampler = original_sampler
        model._prepare_diffusion_stopping_criteria = original_stopping


class FinalReadoutCanvas(JsonCanvas):
    """Denoise values freely; select allowed tokens only from the final logits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.factorized:
            raise ValueError("Final readout currently requires independent aligned token slots")
        self.mask[self.variable_positions] = True

    def select_final(self):
        if self.last_value_logits is None:
            raise ValueError("Must run a denoising step before readout")
        batch_size = self.last_value_logits[0].shape[0]
        output = self.sequences[0].expand(batch_size, -1).clone()
        for position, scores in zip(self.variable_positions, self.last_value_logits):
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite final choice logits")
            choices = torch.tensor(self.columns[position], device=output.device)
            output[:, position] = choices[scores.argmax(-1)]
        return output


def check_constraints(tokenizer, vocab_size, eos):
    from transformers import LogitsProcessorList
    layout = JsonCanvas(tokenizer, routing_outputs(), 32, vocab_size, eos, "cpu")
    assert layout.factorized and layout.variable_positions == [4, 11]
    # Exercise every output, while a forbidden token has an overwhelming score.
    scores = torch.zeros(6, 32, vocab_size)
    scores[..., eos] = 10000
    for row, candidate in enumerate(layout.sequences):
        for position in layout.variable_positions:
            scores[row, position, candidate[position]] = 100
    masked = LogitsProcessorList([layout])(None, scores, cur_step=1)
    decoded = [layout.decode(row)[0] for row in masked.argmax(-1).tolist()]
    assert set(decoded) == set(routing_outputs())
    assert torch.isfinite(torch.distributions.Categorical(logits=masked).entropy()).all()
    readout = FinalReadoutCanvas(tokenizer, routing_outputs(), 32, vocab_size, eos, "cpu")
    free_values = readout(None, scores)
    assert torch.equal(free_values[:, readout.variable_positions], scores[:, readout.variable_positions])
    # Forbidden EOS wins the raw prediction, but final allowed-choice argmax is exact.
    assert (free_values[:, readout.variable_positions].argmax(-1) == eos).all()
    assert set(readout.decode(row)[0] for row in readout.select_final().tolist()) == set(routing_outputs())
    # Compare directly with stock uniform noise under the same RNG seed.
    torch.manual_seed(123)
    expected_noise = torch.randint(vocab_size, (20, 32))
    torch.manual_seed(123)
    actual_noise = layout.initialize_canvas(20, "cpu")
    assert torch.equal(actual_noise[:, layout.variable_positions], expected_noise[:, layout.variable_positions])
    for position, ids in enumerate(layout.columns):
        if len(ids) == 1:
            assert (actual_noise[:, position] == ids[0]).all()
    assert any(actual_noise[0, i].item() not in layout.columns[i] for i in layout.variable_positions)
    for row in layout.initialize_valid_canvas(20, "cpu").tolist():
        layout.decode(row)
    # Correlated alternatives: independent masks would permit forbidden hybrids.
    correlated = [routing_outputs()[0], routing_outputs()[-1]]
    coupled = JsonCanvas(tokenizer, correlated, 32, vocab_size, eos, "cpu")
    assert not coupled.factorized
    for row in coupled(None, scores).argmax(-1).tolist():
        assert coupled.decode(row)[0] in correlated
    # Multi-token and unequal-length values must stay on complete candidate paths.
    longer = [json.dumps({"department": name, "refund_requested": flag})
              for name, flag in itertools.product(
                  ("billing support", "technical escalation team", "sales"), (False, True))]
    extended = JsonCanvas(tokenizer, longer, 32, vocab_size, eos, "cpu")
    assert not extended.factorized
    for row in extended(None, torch.randn_like(scores)).argmax(-1).tolist():
        assert extended.decode(row)[0] in longer
    return {"checks": "passed", "variable_positions": layout.variable_positions,
            "value_tokens": {str(i): [(tokenizer.decode([t]), t) for t in layout.columns[i]]
                             for i in layout.variable_positions}}


class FinalTrieCanvas(JsonCanvas):
    # Structure shared by every complete alternative is fixed. All other
    # positions denoise over the full vocabulary until the final readout.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask[self.variable_positions] = True

    def __call__(self, input_ids, scores):
        self.steps += 1
        self.last_value_logits = [scores[:, p, self.columns[p]].detach().float().clone()
                                  for p in self.variable_positions]
        return scores.masked_fill(~self.mask[None], -torch.inf)

    def select_final(self):
        if self.last_value_logits is None:
            raise ValueError("Run denoising before final readout")
        batch = self.last_value_logits[0].shape[0]
        active = torch.ones((batch, len(self.sequences)), dtype=torch.bool, device=self.sequences.device)
        for pos, logits in zip(self.variable_positions, self.last_value_logits):
            assert torch.isfinite(logits).all()
            tokens = self.sequences[:, pos]
            indices = torch.tensor([self.columns[pos].index(t) for t in tokens.tolist()], device=tokens.device)
            scores = logits[:, indices].masked_fill(~active, -torch.inf)
            chosen = tokens[scores.argmax(-1)]
            active &= tokens[None] == chosen[:, None]
        assert (active.sum(-1) == 1).all()
        return self.sequences[active.long().argmax(-1)]
