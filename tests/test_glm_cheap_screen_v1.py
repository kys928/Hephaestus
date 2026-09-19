from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_glm_cheap_screen_v1 as launcher
import run_glm_cheap_screen_v1 as runner
import validate_glm_cheap_screen_v1 as validator


def load_screen() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_glm_cheap_screen_v1.json").read_text(encoding="utf-8"))


def test_protocol_is_bounded_screening_only() -> None:
    summary = validator.validate()
    assert summary["status"] == "valid"
    assert summary["model_id"] == "zai-org/GLM-4.7-Flash"
    assert summary["revision"] == "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    assert summary["probe_count"] == 5
    assert summary["seed_count"] == 1
    assert summary["total_max_new_tokens"] == 2912
    assert summary["training_performed"] is False
    assert summary["hard_wall_seconds"] <= 1500
    assert summary["max_hourly_usd"] <= 2.25
    assert summary["max_estimated_total_usd"] <= 0.75


def test_launcher_has_no_expensive_gpu_fallback_or_volume() -> None:
    screen = load_screen()
    env = launcher.placeholder_environment("abc123", "glm-cheap-screen-v1-test")
    body = launcher.build_pod_request(screen, name="hephaestus-glm-cheap-screen-v1-test", env=env)
    assert body["gpuCount"] == 1
    assert all("RTX PRO 6000 Blackwell" in gpu for gpu in body["gpuTypeIds"])
    assert not any("H200" in gpu or "B200" in gpu for gpu in body["gpuTypeIds"])
    assert "networkVolumeId" not in body
    assert "volumeMountPath" not in body
    shell = body["dockerStartCmd"][-1]
    assert "run_glm_cheap_screen_v1.py" in shell
    assert "run_diagnostic_scaling_recovery_v1.py" not in shell
    assert "peft" not in shell.casefold()
    assert "torch==2.14.0+cu130" in shell
    assert "download.pytorch.org/whl/cu130" in shell
    assert "sm_120" in shell
    assert "--system-site-packages" not in shell


def test_paid_launch_state_is_explicit_and_requires_user_go() -> None:
    screen = load_screen()
    governance = screen["governance"]
    assert governance["paid_launch_allowed_now"] in (True, False)
    assert governance["paid_launch_requires_explicit_user_go"] is True
    if governance["paid_launch_allowed_now"]:
        assert governance.get("authorization_source") == "user_directive"
        assert str(governance.get("authorization_text") or "").strip()


def test_runner_never_contains_lora_training_surface() -> None:
    source = (ROOT / "scripts/run_glm_cheap_screen_v1.py").read_text(encoding="utf-8").casefold()
    forbidden = ("get_peft_model", "loraconfig", "optimizer.step(", "loss.backward(", "train_role_curve(")
    assert all(token.casefold() not in source for token in forbidden)


def test_screen_disposition_requires_every_probe_and_no_reasoning_exhaustion() -> None:
    complete = [
        {"probe_pass": True, "lane": "fixed_non_thinking", "generation": {"runtime_deadline_hit": False, "finish_reason": "eos"}},
        {"probe_pass": True, "lane": "fixed_non_thinking", "generation": {"runtime_deadline_hit": False, "finish_reason": "eos"}},
        {"probe_pass": True, "lane": "fixed_non_thinking", "generation": {"runtime_deadline_hit": False, "finish_reason": "eos"}},
        {"probe_pass": True, "lane": "reasoning_aware", "generation": {"runtime_deadline_hit": False, "finish_reason": "eos"}},
        {"probe_pass": True, "lane": "reasoning_aware", "generation": {"runtime_deadline_hit": False, "finish_reason": "stopped"}},
    ]
    disposition, advance = runner.screen_disposition(complete, 5)
    assert disposition == "screen_passed_eligible_for_tiny_adaptation_only"
    assert advance is True

    exhausted = [dict(row) for row in complete]
    exhausted[3] = {
        "probe_pass": None,
        "lane": "reasoning_aware",
        "generation": {"runtime_deadline_hit": False, "finish_reason": "max_tokens"},
    }
    disposition, advance = runner.screen_disposition(exhausted, 5)
    assert disposition == "screen_inconclusive_reasoning_budget_exhausted"
    assert advance is False