"""Real CPU role-conditioned weight updates; never certifies production agents."""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict

import pytest

from hephaestus.llm.cpu_research_smoke import (
    ROLES, VERDICTS, infer_cpu_research_role, train_cpu_research_roles,
)


@pytest.mark.real_cpu
def test_five_cpu_research_roles_use_actual_trained_weights(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("requires actual CPU PyTorch")
    checkpoint = train_cpu_research_roles(tmp_path / "cpu-model" / "role_model.pt")
    assert checkpoint.optimizer_steps >= 1
    assert checkpoint.final_loss < checkpoint.initial_loss
    assert checkpoint.training_accuracy >= 0.975
    scenarios = {
        "controller": dict(evidence_present=True, provenance_verified=True,
                           contradiction=False, approval_present=True),
        "diagnosis": dict(evidence_present=True, provenance_verified=True,
                          contradiction=False, approval_present=False),
        "planner": dict(evidence_present=True, provenance_verified=False,
                        contradiction=False, approval_present=True),
        "evaluator": dict(evidence_present=True, provenance_verified=True,
                          contradiction=True, approval_present=False),
        "judge": dict(evidence_present=True, provenance_verified=True,
                      contradiction=False, approval_present=False),
    }
    outcomes = [
        infer_cpu_research_role(checkpoint, role=role, **scenarios[role])
        for role in ROLES
    ]
    assert len(outcomes) == 5
    assert [doc["selection"] for doc in outcomes] == [
        "proceed_with_review", "proceed_with_review", "needs_evidence",
        "reject_observed", "needs_evidence",
    ]
    assert all(len(doc["logits"]) == len(VERDICTS) for doc in outcomes)
    assert all(not doc["is_certified_model"] and not doc["action_authorized"]
               for doc in outcomes)
    assert len({doc["checkpoint_sha256"] for doc in outcomes}) == 1

    path = os.environ.get("HEPHAESTUS_CPU_ROLE_SMOKE_REPORT")
    if path:
        from pathlib import Path
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "cpu-five-role-measured-smoke.v1",
            "scope": "synthetic_feature_role_classifier_cpu_research_only",
            "actual_autograd_training": True,
            "frozen_production_role_models_used": False,
            "full_45_operation_campaign_executed": False,
            "production_certification": "not_certified",
            "promotion_allowed": False,
            "training": {k: str(v) if k == "path" else v
                         for k, v in asdict(checkpoint).items()},
            "inference": outcomes,
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
