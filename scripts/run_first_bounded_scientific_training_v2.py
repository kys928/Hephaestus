#!/usr/bin/env python3
"""Execute the first bounded scientific driver with the reviewed v2 pilot recipe."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).with_name("run_first_bounded_scientific_training.py")
spec = importlib.util.spec_from_file_location("hephaestus_first_training_driver_v1", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load first scientific training driver")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

_original_build_proposal = module.build_proposal
_original_result_for = module.result_for


def _build_proposal(contract: Path, evidence: Path, model: Path, tokenizer: Path, special: dict[str, int]):
    proposal = _original_build_proposal(contract, evidence, model, tokenizer, special)
    constraints = dict(proposal.training_constraints)
    constraints.update(
        {
            "max_steps": 100,
            "warmup_steps": 10,
            "checkpoint_every_steps": 100,
            "logging_every_steps": 5,
            "max_total_tokens": 20_000_000,
        }
    )
    return replace(
        proposal,
        training_constraints=constraints,
        metadata={
            **proposal.metadata,
            "training_recipe_ref": "first-bounded-scientific-training.v2",
            "recipe_revision_note": "100-step bounded lifecycle proof with one finalized checkpoint",
        },
    )


def _result_for(proposal: Any, handle: Any, sink: Any) -> dict[str, object]:
    result = _original_result_for(proposal, handle, sink)
    recipe = result.get("bounded_recipe")
    if isinstance(recipe, dict):
        recipe.update(
            {
                "max_steps": 100,
                "warmup_steps": 10,
                "checkpoint_every_steps": 100,
                "max_total_tokens": 20_000_000,
                "recipe_version": "first-bounded-scientific-training.v2",
            }
        )
    return result


module.build_proposal = _build_proposal
module.result_for = _result_for

if __name__ == "__main__":
    raise SystemExit(module.main())
