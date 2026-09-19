from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_diagnostic_scaling_recovery_v1.py"
spec = importlib.util.spec_from_file_location("diagnostic_scaling_recovery_runtime", SCRIPT)
assert spec and spec.loader
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def contract() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json").read_text(encoding="utf-8"))


def candidate(model_id: str = "Qwen/Qwen3-14B") -> dict:
    return next(row for row in contract()["candidates"] if row["model_id"] == model_id)


def test_projection_scores_text_after_last_reasoning_close_delimiter():
    raw = "<think>first</think> junk <think>second</think> {\"decision\": \"ok\"}"
    assert runtime.project_reasoning(raw, contract(), "reasoning_aware") == '{"decision": "ok"}'


def test_fixed_lane_does_not_strip_reasoning_text():
    raw = "<think>x</think> answer"
    assert runtime.project_reasoning(raw, contract(), "fixed_non_thinking") == raw


def test_fixed_lane_eligibility_is_explicit():
    assert runtime.lane_eligible(candidate("Qwen/Qwen3-14B"), "fixed_non_thinking") is True
    assert runtime.lane_eligible(candidate("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"), "fixed_non_thinking") is False
    assert runtime.lane_eligible(candidate("Qwen/Qwen3-30B-A3B-Thinking-2507"), "fixed_non_thinking") is False


def test_adaptive_generation_retries_only_token_budget_exhaustion(monkeypatch):
    calls = []
    def fake_generate(*args, max_new_tokens, **kwargs):
        calls.append(max_new_tokens)
        return {
            "output": '<think>done</think>{"decision":"x","action":"y","primary_variable":"z","confidence":0.5,"evidence_refs":[],"uncertainties":[],"rationale":"r"}',
            "finish_reason": "eos",
            "generated_tokens": 1200,
        }

    monkeypatch.setattr(runtime, "generate_once", fake_generate)
    result = runtime.adaptive_generate(
        object(), object(), "p", candidate=candidate(), lane="reasoning_aware", seed=11,
        token_ladder=[1024, 2048, 4096], contract=contract(), require_schema=True,
    )
    assert calls == [4096]
    assert result["budget_retry_count"] == 1
    assert result["physical_generation_count"] == 1
    assert result["budget_exhausted"] is False
    assert result["final_schema_complete"] is True


def test_eos_malformed_json_retries_before_becoming_inconclusive(monkeypatch):
    calls = []

    def fake_generate(*args, max_new_tokens, **kwargs):
        calls.append(max_new_tokens)
        return {"output": "not json", "finish_reason": "eos", "generated_tokens": 14}

    monkeypatch.setattr(runtime, "generate_once", fake_generate)
    result = runtime.adaptive_generate(
        object(), object(), "p", candidate=candidate(), lane="reasoning_aware", seed=11,
        token_ladder=[1024, 2048, 4096], contract=contract(), require_schema=True,
    )
    assert calls == [4096]
    assert result["budget_retry_count"] == 2
    assert result["physical_generation_count"] == 1
    assert result["budget_exhausted"] is True
    assert result["final_schema_complete"] is False


def test_terminal_budget_exhaustion_is_inconclusive_signal(monkeypatch):
    calls = []

    def fake_generate(*args, max_new_tokens, **kwargs):
        calls.append(max_new_tokens)
        return {"output": "<think>unfinished", "finish_reason": "max_tokens", "generated_tokens": max_new_tokens}

    monkeypatch.setattr(runtime, "generate_once", fake_generate)
    result = runtime.adaptive_generate(
        object(), object(), "p", candidate=candidate(), lane="reasoning_aware", seed=11,
        token_ladder=[1024, 2048], contract=contract(), require_schema=True,
    )
    assert calls == [2048]
    assert result["budget_exhausted"] is True
    assert result["physical_generation_count"] == 1
    assert result["budget_retry_count"] == 1


def test_lane_points_exclude_nonclaimable_reasoning_results():
    points = [
        {
            "optimizer_steps": 0,
            "approx_epochs": 0.0,
            "training": {"supervised_tokens_cumulative": 0},
            "lanes": {
                "reasoning_aware": {
                    "status": "inconclusive_reasoning_budget_exhausted",
                    "role": {"quality_100": 95.0},
                    "semantic": {"mean_score": 1.0},
                    "safety": {"claimable": False, "interface_ok": True, "strict_safe": True, "bounded_safe": True},
                }
            },
        }
    ]
    assert runtime.lane_points(points, "reasoning_aware") == []


def test_recovery_driver_retains_paid_launch_governance_guard():
    payload = contract()
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if contract["governance"].get("paid_launch_allowed_now") is not True:' in source
    assert "paid Diagnostic Scaling Recovery launch is not authorized" in source
    assert payload["governance"]["paid_launch_requires_new_explicit_user_go"] is True


def test_recovery_generation_reenables_kv_cache_for_inference():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "use_cache=True" in source
    payload = contract()
    assert payload["execution"]["inference_runtime"]["use_kv_cache"] is True


def test_recovery_streams_incremental_s3_progress():
    source = SCRIPT.read_text(encoding="utf-8")
    payload = contract()
    assert payload["execution"]["progress_checkpointing"]["enabled"] is True
    assert "DIAGNOSTIC_SCALING_RECOVERY_CHECKPOINT_JSON" in source
    assert "DIAGNOSTIC_SCALING_RECOVERY_BASELINE_CHECKPOINT_JSON" in source
    assert "DIAGNOSTIC_SCALING_RECOVERY_ROLE_CHECKPOINT_JSON" in source
    assert "sync_tree_verified(client, adapter_dir" in source


def test_adaptive_budget_execution_is_single_physical_decode():
    payload = contract()
    settings = payload["execution"]["inference_runtime"]
    assert settings["adaptive_budget_execution"] == "single_physical_terminal_cap_with_virtual_retry_accounting"
    assert settings["scientific_geometry_changed"] is False


def test_recovery_reuses_completed_baseline_and_roles():
    payload = contract()
    settings = payload["execution"]["progress_checkpointing"]
    source = SCRIPT.read_text(encoding="utf-8")
    assert settings["resume_baseline"] is True
    assert settings["resume_completed_roles"] is True
    assert "DIAGNOSTIC_SCALING_RECOVERY_BASELINE_REUSED_JSON" in source
    assert "DIAGNOSTIC_SCALING_RECOVERY_ROLE_REUSED_JSON" in source
    assert "validate_reused_role_checkpoint" in source
