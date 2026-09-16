"""One-canvas categorical decisions; deliberately not a general JSON grammar."""
import torch


class ChoiceCanvasProcessor:
    """Restrict the first N positions to categorical labels; force EOS afterwards.

    This processor is per request. Questions on one canvas can attend to each
    other, so this does NOT reproduce Jev's question-isolation guarantee.
    """

    def __init__(self, allowed_ids, canvas_length, vocab_size, eos_token_id, device):
        if not 0 < len(allowed_ids) < canvas_length:
            raise ValueError("Need 1..canvas_length-1 choices (reserve an EOS slot)")
        if not 0 <= eos_token_id < vocab_size:
            raise ValueError("Invalid EOS token")
        self.allowed_ids = tuple(tuple(ids) for ids in allowed_ids)
        self.mask = torch.zeros((canvas_length, vocab_size), dtype=torch.bool, device=device)
        for position, ids in enumerate(self.allowed_ids):
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("Choices must be nonempty and token IDs unique")
            if any(i < 0 or i >= vocab_size or i == eos_token_id for i in ids):
                raise ValueError("Invalid choice token")
            self.mask[position, list(ids)] = True
        self.mask[len(allowed_ids):, eos_token_id] = True
        self.steps = 0
        self.last_choice_logits = None

    def __call__(self, input_ids, scores):
        if scores.ndim != 3 or scores.shape[1:] != self.mask.shape:
            raise ValueError("Expected [batch, canvas_length, vocab_size] diffusion logits")
        # Record only small categorical slices, before the generation temperature.
        self.last_choice_logits = [
            scores[:, i, list(ids)].detach().float().clone()
            for i, ids in enumerate(self.allowed_ids)
        ]
        self.steps += 1
        return scores.masked_fill(~self.mask[None], -torch.inf)

    def decode(self, generated_ids, choices):
        if len(choices) != len(self.allowed_ids):
            raise ValueError("Wrong number of choices")
        if len(generated_ids) < len(choices):
            raise ValueError("Incomplete result")
        output = []
        for token_id, ids, values in zip(generated_ids, self.allowed_ids, choices):
            if len(values) != len(ids):
                raise ValueError("Choices and labels disagree")
            if token_id not in ids:
                raise ValueError("Model output violates the choice constraint")
            output.append(values[ids.index(token_id)])
        return output


def label_tokens(tokenizer, count):
    """Use verified single-token labels; user values can be arbitrary strings."""
    if not 1 <= count <= 26:
        raise ValueError("Prototype supports 1..26 alternatives per question")
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count])
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("Tokenizer does not encode these labels as single tokens")
    flat = [tokens[0] for tokens in ids]
    if len(set(flat)) != count or any(i in tokenizer.all_special_ids for i in flat):
        raise ValueError("Labels must have unique, ordinary token IDs")
    return labels, flat
