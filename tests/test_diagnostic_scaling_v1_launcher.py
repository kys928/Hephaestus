from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_diagnostic_scaling_v1.py"
RUNNER = ROOT / "scripts/run_diagnostic_scaling_v1.py"

launcher_spec = importlib.util.spec_from_file_location("diagnostic_scaling_launcher", LAUNCHER)
assert launcher_spec and launcher_spec.loader
launcher = importlib.util.module_from_spec(launcher_spec)
launcher_spec.loader.exec_module(launcher)

runner_spec = importlib.util.spec_from_file_location("diagnostic_scaling_runner", RUNNER)
assert runner_spec and runner_spec.loader
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def contract() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json").read_text())


def test_all_rendered_requests_are_ephemeral_and_immutable():
    payload = contract()
    for candidate in payload["candidates"]:
        rendered = launcher.render_only(
            candidate["model_id"],
            repo_sha="f" * 40,
            run_id="diagnostic-scaling-v1-dry-run",
        )
        assert rendered["status"] == "rendered_not_launched"
        assert rendered["launch_authorized"] is False
        request = rendered["request"]
        assert "networkVolumeId" not in request
        assert "volumeMountPath" not in request
        assert request["containerDiskInGb"] == 300
        assert request["imageName"] == "pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime"
        assert request["cloudType"] == "SECURE"
        assert request["dataCenterIds"] == ["EU-CZ-1"]
        assert request["env"]["HEPHAESTUS_REPO_SHA"] == "f" * 40
        assert request["env"]["RUNPOD_S3_ACCESS_KEY_ID"] == "<redacted>"
        assert request["env"]["RUNPOD_S3_SECRET_ACCESS_KEY"] == "<redacted>"
        shell = request["dockerStartCmd"][-1]
        assert 'git checkout --detach "$HEPHAESTUS_REPO_SHA"' in shell
        assert "scripts/run_diagnostic_scaling_v1.py" in shell


def test_large_moe_candidates_are_h200_only():
    payload = contract()
    for candidate in payload["candidates"]:
        ids = launcher.gpu_ids(payload, candidate)
        if candidate["minimum_gpu_memory_gib"] >= 140:
            assert ids == ["NVIDIA H200"]
        else:
            assert "NVIDIA H200" in ids
            assert any("H100" in value or "A100" in value for value in ids)


def test_network_volume_injection_is_rejected():
    payload = contract()
    candidate = payload["candidates"][0]
    env = launcher.placeholder_environment(
        repo_sha="e" * 40,
        run_id="dry",
        execution_id="dry-model",
        attempt=1,
        model_id=candidate["model_id"],
    )
    request = launcher.build_pod_request(payload, candidate, name="dry", env=env)
    request = copy.deepcopy(request)
    request["networkVolumeId"] = "forbidden"
    with pytest.raises(ValueError, match="must not attach"):
        launcher.validate_pod_request(payload, candidate, request)


def test_reasoning_projection_preserves_raw_fallback_and_scores_final_answer():
    payload = contract()
    raw = '<think>private scratch</think> {"decision":"ok"}'
    assert runner.project_reasoning(raw, payload) == '{"decision":"ok"}'
    plain = '{"decision":"plain"}'
    assert runner.project_reasoning(plain, payload) == plain


def test_runtime_source_contains_governed_moe_target_parameter_path():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'lora_kwargs["target_parameters"] = target_parameters' in source
    assert 'lora_kwargs["rank_pattern"] = rank_pattern' in source
    assert "router unexpectedly became trainable" in source
