from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_diagnostic_scaling_recovery_v1.py"
spec = importlib.util.spec_from_file_location("diagnostic_scaling_recovery_contract", SCRIPT)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def contract() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json").read_text(encoding="utf-8"))


def test_recovery_contract_is_valid_in_current_authorization_state():
    payload = contract()
    evidence = validator.validate_contract(payload, ROOT)
    assert evidence["status"] == "recovery_contract_valid"
    assert evidence["candidate_count"] == 4
    assert evidence["paid_launch_allowed_now"] is bool(payload["governance"]["paid_launch_allowed_now"])
    assert evidence["authorization_state"] in {"blocked_pre_user_go", "authorized_by_user"}
    assert evidence["frozen_eval_mutated"] is False
    assert evidence["fixed_non_thinking_eligible"] == ["Qwen/Qwen3-14B", "zai-org/GLM-4.7-Flash"]
    assert len(evidence["reasoning_aware_eligible"]) == 4


def test_recovery_preserves_training_geometry():
    payload = contract()
    payload["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="training geometry drifted"):
        validator.validate_contract(payload, ROOT)


def test_fixed_lane_cannot_enable_thinking():
    payload = copy.deepcopy(contract())
    payload["candidates"][0]["fixed_non_thinking"]["chat_template_kwargs"]["enable_thinking"] = True
    with pytest.raises(ValueError, match="hard non-thinking switch"):
        validator.validate_contract(payload, ROOT)


def test_thinking_only_model_cannot_receive_synthetic_fixed_lane_score():
    payload = copy.deepcopy(contract())
    payload["candidates"][3]["fixed_non_thinking"] = {
        "eligible": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with pytest.raises(ValueError, match="eligibility drifted"):
        validator.validate_contract(payload, ROOT)


def test_reasoning_lane_cannot_score_incomplete_answers():
    payload = copy.deepcopy(contract())
    payload["evaluation_lanes"]["reasoning_aware"]["score_incomplete_final_answer"] = True
    with pytest.raises(ValueError, match="retry incomplete answers"):
        validator.validate_contract(payload, ROOT)


def test_reasoning_budget_exhaustion_must_be_inconclusive_not_floor_score():
    payload = copy.deepcopy(contract())
    payload["evaluation_lanes"]["reasoning_aware"]["budget_exhaustion_policy"] = "score_zero"
    with pytest.raises(ValueError, match="budget exhaustion policy"):
        validator.validate_contract(payload, ROOT)


def test_qwen30_reasoning_budget_cannot_regress_to_old_256_token_cap():
    payload = copy.deepcopy(contract())
    payload["candidates"][3]["reasoning_aware"]["topology_token_ladder"] = [512, 1024, 2048]
    with pytest.raises(ValueError, match="terminal budget"):
        validator.validate_contract(payload, ROOT)


def test_paid_launch_always_requires_explicit_user_go_guard():
    payload = copy.deepcopy(contract())
    payload["governance"]["paid_launch_requires_new_explicit_user_go"] = False
    with pytest.raises(ValueError, match="explicit user go"):
        validator.validate_contract(payload, ROOT)


def test_authorized_paid_launch_requires_exact_user_provenance():
    payload = copy.deepcopy(contract())
    if payload["governance"]["paid_launch_allowed_now"] is not True:
        pytest.skip("current contract is still pre-launch")
    payload["governance"]["approval_source"] = "invalid"
    with pytest.raises(ValueError, match="approval source"):
        validator.validate_contract(payload, ROOT)


def test_cross_lane_ranking_is_forbidden():
    payload = copy.deepcopy(contract())
    payload["evaluation_lanes"]["fixed_non_thinking"]["cross_lane_ranking_allowed"] = True
    with pytest.raises(ValueError, match="cross-lane ranking"):
        validator.validate_contract(payload, ROOT)


def test_reasoning_lanes_have_sufficient_terminal_compute_and_startup_watchdog():
    payload = contract()
    for row in payload["candidates"]:
        assert row["reasoning_aware"]["topology_token_ladder"][-1] >= 32768
        assert row["reasoning_aware"]["semantic_token_ladder"][-1] >= 8192
    assert 300 <= payload["execution"]["silent_container_start_timeout_seconds"] <= 1800
    assert payload["execution"]["image"] == "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
    assert payload["execution"]["preferred_gpu_for_30b"] == ["NVIDIA H200"]
    assert payload["execution"]["cuda_runtime_recovery"]["fail_fast_cuda_preflight_before_model_download"] is True
    assert payload["execution"]["cuda_runtime_recovery"]["gpu_memory_floor_tolerance_gib"] == 0.5
