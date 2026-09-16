"""Compositional finite-field JSON canvas, without Cartesian enumeration."""

import json
import torch


def compile_fields(tokenizer, fields, capacity=256):
    space = tokenizer.encode(" ", add_special_tokens=False)
    if len(space) != 1 or tokenizer.decode(space) != " ":
        raise ValueError("Need a single whitespace token for value padding")
    template = tokenizer.encode("{\n", add_special_tokens=False)
    slots = []
    for index, (key, options) in enumerate(fields.items()):
        if index:
            template += tokenizer.encode(",\n", add_special_tokens=False)
        choices = [
            tokenizer.encode(
                json.dumps(key) + ": " + json.dumps(v), add_special_tokens=False
            )
            for v in options
        ]
        width = max(map(len, choices))
        choices = [c + space * (width - len(c)) for c in choices]
        slots.append((len(template), choices))
        template += choices[0]
    template += tokenizer.encode("\n}", add_special_tokens=False)
    if len(template) >= capacity:
        raise ValueError("Fields exceed canvas capacity")
    # Validate concatenated tokens rather than assuming tokenizer boundaries.
    expected = {k: v[0] for k, v in fields.items()}
    if json.loads(tokenizer.decode(template)) != expected:
        raise ValueError("Template does not decode to the expected JSON")
    for (key, options), (offset, choices) in zip(fields.items(), slots):
        for option, choice in zip(options, choices):
            ids = template.copy()
            ids[offset : offset + len(choice)] = choice
            if json.loads(tokenizer.decode(ids)) != dict(expected, **{key: option}):
                raise ValueError("Candidate tokenization mismatch")
    return template, slots


class FieldCanvas:
    def __init__(self, tokenizer, fields, capacity, vocab_size, eos, device):
        template, slots = compile_fields(tokenizer, fields, capacity)
        self.tokenizer, self.fields, self.eos = tokenizer, fields, eos
        self.template = torch.tensor(
            template + [eos] * (capacity - len(template)), device=device
        )
        self.mask = torch.zeros((capacity, vocab_size), dtype=torch.bool, device=device)
        self.mask.scatter_(1, self.template[:, None], True)
        self.slots = []
        self.variable_positions = []
        for offset, choices in slots:
            candidates = torch.tensor(choices, device=device)
            variables = []
            for j in range(candidates.shape[1]):
                allowed = sorted(set(c[j] for c in choices))
                if len(allowed) > 1:
                    p = offset + j
                    self.mask[p] = True
                    self.variable_positions.append(p)
                    variables.append((p, j, allowed))
            self.slots.append((offset, candidates, variables))
        self.steps = 0
        self.last = {}

    def initialize_canvas(self, batch_size, device):
        canvas = self.template.expand(batch_size, -1).clone()
        noise = torch.randint(self.mask.shape[1], canvas.shape, device=device)
        canvas[:, self.variable_positions] = noise[:, self.variable_positions]
        return canvas

    def __call__(self, input_ids, scores):
        self.steps += 1
        self.last = {
            p: scores[:, p, allowed].detach().float().clone()
            for _, _, variables in self.slots
            for p, _, allowed in variables
        }
        return scores.masked_fill(~self.mask[None], -torch.inf)

    def select_final(self):
        batch = next(iter(self.last.values())).shape[0]
        output = self.template.expand(batch, -1).clone()
        for offset, candidates, variables in self.slots:
            active = torch.ones(
                (batch, len(candidates)), dtype=torch.bool, device=output.device
            )
            for p, j, allowed in variables:
                logits = self.last[p]
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite logits")
                tokens = candidates[:, j]
                indices = torch.tensor(
                    [allowed.index(t) for t in tokens.tolist()], device=output.device
                )
                chosen = tokens[
                    logits[:, indices].masked_fill(~active, -torch.inf).argmax(-1)
                ]
                active &= tokens[None] == chosen[:, None]
            selected = candidates[active.long().argmax(-1)]
            output[:, offset : offset + candidates.shape[1]] = selected
        return output

    def decode(self, ids):
        text = self.tokenizer.decode(ids[: ids.index(self.eos)])
        value = json.loads(text)
        if list(value) != list(self.fields):
            raise ValueError("Incorrect keys")
        for k, v in value.items():
            if json.dumps(v) not in [json.dumps(x) for x in self.fields[k]]:
                raise ValueError("Invalid value")
        return text, value


class BatchCanvas:
    def __init__(self, layouts):
        self.layouts = layouts

    def initialize_canvas(self, batch_size, device):
        assert batch_size == len(self.layouts)
        rows = []
        for layout in self.layouts:
            torch.manual_seed(0)
            rows.append(layout.initialize_canvas(1, device))
        return torch.cat(rows)

    def __call__(self, input_ids, scores):
        return torch.cat(
            [layout(None, scores[i : i + 1]) for i, layout in enumerate(self.layouts)]
        )
