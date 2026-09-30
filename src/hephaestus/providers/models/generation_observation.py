"""Passive token observations; generated text and frozen evaluator inputs are unchanged."""
from __future__ import annotations


def token_termination(token_ids: list[int], eos_token_id: int, budget: int) -> dict[str, object]:
    try:
        end = token_ids.index(eos_token_id) + 1
        return {"eos_observed": True, "actual_generated_tokens": end, "actual_finish_reason": "eos"}
    except ValueError:
        return {"eos_observed": False, "actual_generated_tokens": len(token_ids),
                "actual_finish_reason": "length" if len(token_ids) >= budget else "unknown"}


class GenerationObserver:
    def __init__(self, model):
        self.model = model
        self.observations = []

    def generate(self, **kwargs):
        generated = self.model.generate(**kwargs)
        width = int(kwargs["input_ids"].shape[1])
        for row in generated:
            self.observations.append(token_termination(
                row[width:].tolist(), int(kwargs["eos_token_id"]), int(kwargs["max_new_tokens"])))
        return generated
