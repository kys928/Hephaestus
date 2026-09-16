from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_diagnostic_scaling_v1_contract.py"
spec = importlib.util.spec_from_file_location("diagnostic_scaling_contract", SCRIPT)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def contract() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json").read_text(encoding="utf-8"))


def test_contract_matches_frozen_test3_geometry_and_governance():
    evidence = validator.validate_contract(contract(), ROOT)
    assert evidence["status"] == "local_contract_valid"
    assert evidence["candidate_count"] == 4
    assert evidence["roles"] == ["diagnosis", "controller"]
    assert evidence["dose_optimizer_steps"] == [3, 6, 12, 24, 36, 48]
    assert evidence["ephemeral_only"] is True
    assert evidence["moe_target_parameters_required"] is True
    assert evidence["promotion_allowed"] is False


def test_mutable_or_wrong_revision_is_rejected():
    payload = contract()
    payload["candidates"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="candidate identities/revisions drifted"):
        validator.validate_contract(payload, ROOT)


def test_network_volume_attachment_is_rejected():
    payload = copy.deepcopy(contract())
    payload["execution"]["network_volume_attached"] = True
    with pytest.raises(ValueError, match="ephemeral"):
        validator.validate_contract(payload, ROOT)


def test_test3_training_geometry_drift_is_rejected():
    payload = copy.deepcopy(contract())
    payload["training"]["learning_rate"] = 0.0001
    with pytest.raises(ValueError, match="training geometry drifted"):
        validator.validate_contract(payload, ROOT)


def test_reasoning_projection_must_preserve_raw_generation():
    payload = copy.deepcopy(contract())
    payload["evaluation"]["reasoning_projection"]["preserve_raw_generation"] = False
    with pytest.raises(ValueError, match="reasoning projection"):
        validator.validate_contract(payload, ROOT)


def test_moe_routed_experts_must_use_target_parameters():
    payload = copy.deepcopy(contract())
    payload["training"]["target_policy"]["qwen3_moe"]["target_parameters"] = []
    with pytest.raises(ValueError, match="MoE target geometry"):
        validator.validate_contract(payload, ROOT)


def test_moe_expert_rank_budget_cannot_silently_expand():
    payload = copy.deepcopy(contract())
    payload["training"]["target_policy"]["glm4_moe_lite"]["expert_rank"] = 8
    with pytest.raises(ValueError, match="MoE target geometry"):
        validator.validate_contract(payload, ROOT)


def test_router_must_stay_frozen():
    payload = copy.deepcopy(contract())
    payload["training"]["router_trainable"] = True
    with pytest.raises(ValueError, match="routers"):
        validator.validate_contract(payload, ROOT)


def test_governance_cannot_enable_promotion():
    payload = copy.deepcopy(contract())
    payload["governance"]["promotion_allowed"] = True
    with pytest.raises(ValueError, match="governance boundary"):
        validator.validate_contract(payload, ROOT)
