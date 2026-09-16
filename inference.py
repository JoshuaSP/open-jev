"""Reusable finite-JSON diffusion inference; requires the inference extra."""

import itertools
import json
import time


def prepare(request, enumerate_candidates=True):
    """Validate a document/questions request before allocating GPU resources."""
    document, questions = request["document"], request["questions"]
    if (
        not isinstance(document, str)
        or not isinstance(questions, dict)
        or not questions
    ):
        raise ValueError("Expected a document string and a nonempty questions object")
    count = 1
    for key, question in questions.items():
        if not isinstance(key, str) or not isinstance(question.get("question"), str):
            raise ValueError("Each field requires a question string")
        options = question["options"]
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("Each field needs at least two alternatives")
        if any(type(v) not in (str, int, bool) for v in options):
            raise ValueError("Supported alternatives: strings, integers, booleans")
        if len({json.dumps(v) for v in options}) != len(options):
            raise ValueError("Duplicate alternatives")
        count *= len(options)
    if enumerate_candidates and count > 4096:
        raise ValueError("Finite-language prototype limit: 4096 complete alternatives")
    alternatives = (
        [
            json.dumps(dict(zip(questions, values)))
            for values in itertools.product(*(q["options"] for q in questions.values()))
        ]
        if enumerate_candidates
        else {k: q["options"] for k, q in questions.items()}
    )
    instructions = "Answer each question using its allowed values. Return only a JSON object with these keys in this order.\n"
    instructions += "\n".join(
        f"{json.dumps(k)}: {q['question']}\nAllowed values: {json.dumps(q['options'])}"
        for k, q in questions.items()
    )
    prompt = (
        "<user_text>\n"
        + document
        + "\n</user_text>\n\n<instructions>\n"
        + instructions
        + "\n</instructions>"
    )
    return prompt, alternatives


class DiffusionHarness:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        processor.tokenizer.padding_side = "left"

    @classmethod
    def load(cls, token=None):
        import torch
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        from modal_app import MODEL, REVISION

        processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, token=token)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            MODEL,
            revision=REVISION,
            token=token,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        return cls(model, processor)

    def predict(self, requests, steps=1, seed=0):
        """Batch requests sharing an identical question schema; propagate OOM."""
        import torch
        from transformers import LogitsProcessorList
        from json_canvas import constrained_sampler
        from parallel_canvas import FieldCanvas

        if not requests or steps < 1:
            raise ValueError("Need requests and a positive step count")
        prepared = [prepare(r, enumerate_candidates=False) for r in requests]
        if any(r["questions"] != requests[0]["questions"] for r in requests):
            raise ValueError("Batch requests must share the same question schema")
        model, processor = self.model, self.processor
        eos = model.generation_config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        layout = FieldCanvas(
            processor.tokenizer,
            prepared[0][1],
            model.config.canvas_length,
            model.config.text_config.vocab_size,
            eos,
            model.device,
        )
        torch.manual_seed(seed)
        initial = layout.initialize_canvas(1, model.device)
        layout.initialize_canvas = lambda batch_size, device: initial.expand(
            batch_size, -1
        ).clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = processor.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt, _ in prepared],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(model.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), constrained_sampler(model, layout):
            model.generate(
                **inputs,
                max_new_tokens=model.config.canvas_length,
                max_denoising_steps=steps,
                cache_implementation="dynamic",
                logits_processor=LogitsProcessorList([layout]),
            )
        answers = [layout.decode(ids)[1] for ids in layout.select_final().tolist()]
        torch.cuda.synchronize()
        return dict(
            answers=answers,
            steps=layout.steps,
            seed=seed,
            request_seconds=time.perf_counter() - start,
            input_tokens=int(inputs["attention_mask"].sum()),
            padded_input_tokens=int(inputs["input_ids"].numel()),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
