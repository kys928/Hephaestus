#!/usr/bin/env python3
"""Run the controlled semantic evaluation with an explicit baseline-ref projection.

The frozen Planner proposal records its baseline as ``run://<run-id>`` for
cross-component provenance. ExperimentEvaluationService compares concrete
TrainingRunHandle IDs. This adapter verifies the frozen form through the base
loader, then projects only the runtime comparison field to the exact run ID.
"""
from __future__ import annotations

from dataclasses import replace

import run_second_controlled_semantic_evaluation as base

_original_load = base.load_frozen_proposal


def _load_runtime_proposal():
    proposal = _original_load()
    source_ref = proposal.baseline_ref
    return replace(
        proposal,
        baseline_ref=base.BASELINE_RUN_ID,
        metadata={
            **proposal.metadata,
            "source_baseline_ref": source_ref,
            "runtime_baseline_run_id": base.BASELINE_RUN_ID,
            "baseline_ref_projection": "run_uri_to_training_run_handle_id",
        },
    )


base.load_frozen_proposal = _load_runtime_proposal


if __name__ == "__main__":
    raise SystemExit(base.main())
