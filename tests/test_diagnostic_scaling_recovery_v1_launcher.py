from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/launch_diagnostic_scaling_recovery_v1.py"
spec = importlib.util.spec_from_file_location("diagnostic_scaling_recovery_launcher", SCRIPT)
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def contract() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json").read_text(encoding="utf-8"))


def candidate(model_id: str = "Qwen/Qwen3-14B") -> dict:
    return next(row for row in contract()["candidates"] if row["model_id"] == model_id)


def test_render_only_never_executes_paid_launch():
    payload = contract()
    rendered = launcher.render_only("Qwen/Qwen3-14B", repo_sha="1" * 40, run_id="dry")
    assert rendered["status"] == "rendered_not_launched"
    assert rendered["launch_authorized"] is False
    assert rendered["paid_launch_allowed_now"] is bool(payload["governance"]["paid_launch_allowed_now"])
    request = rendered["request"]
    assert "networkVolumeId" not in request
    assert "volumeMountPath" not in request
    assert request["gpuCount"] == 1
    assert request["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert request["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    shell = request["dockerStartCmd"][-1]
    assert request["imageName"] == "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
    assert "HEPHAESTUS_CUDA_PREFLIGHT_JSON" in shell
    assert "HEPHAESTUS_CUDA_PREFLIGHT_OK" in shell
    assert shell.index("HEPHAESTUS_CUDA_PREFLIGHT_JSON") < shell.index("apt-get update")
    assert "scripts/run_diagnostic_scaling_recovery_v1.py" in shell
    assert "git checkout --detach \"$HEPHAESTUS_REPO_SHA\"" in shell


def test_paid_execute_obeys_committed_contract_and_second_env_lock():
    payload = contract()
    if payload["governance"]["paid_launch_allowed_now"] is True:
        launcher.paid_launch_gate(payload, execute=True, authorization_env="YES")
    else:
        with pytest.raises(RuntimeError, match="blocked by the committed recovery contract"):
            launcher.paid_launch_gate(payload, execute=True, authorization_env="YES")


def test_paid_execute_still_requires_new_authorization_env():
    payload = copy.deepcopy(contract())
    payload["governance"]["paid_launch_allowed_now"] = True
    with pytest.raises(RuntimeError, match="without HEPHAESTUS_DIAGNOSTIC_SCALING_RECOVERY_LAUNCH_AUTHORIZED=YES"):
        launcher.paid_launch_gate(payload, execute=True, authorization_env="NO")
    launcher.paid_launch_gate(payload, execute=True, authorization_env="YES")


def test_pod_request_rejects_network_volume_attachment():
    payload = contract()
    row = candidate()
    body = launcher.build_pod_request(
        payload,
        row,
        name="test",
        env=launcher.placeholder_environment(repo_sha="1" * 40, run_id="dry", execution_id="dry-model", attempt=1, model_id=row["model_id"]),
    )
    body["networkVolumeId"] = "forbidden"
    with pytest.raises(ValueError, match="no Network Volume"):
        launcher.validate_pod_request(payload, row, body)


def test_30b_gpu_allowlist_is_large_memory_only():
    payload = contract()
    row = next(r for r in payload["candidates"] if r["model_id"] == "Qwen/Qwen3-30B-A3B-Thinking-2507")
    assert launcher.gpu_ids(payload, row) == ["NVIDIA H200"]


def test_wait_terminal_fails_immediately_on_exited_pod(monkeypatch):
    class Client:
        pass

    monkeypatch.setattr(launcher.storage, "maybe_read_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.storage, "pod_snapshot", lambda *args, **kwargs: {"desiredStatus": "EXITED", "id": "pod-x"})
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    with pytest.raises(launcher.PodExitedWithoutTerminal) as exc:
        launcher.wait_terminal(Client(), object(), execution_id="exec", attempt=1, pod_id="pod-x")
    assert exc.value.snapshot["desiredStatus"] == "EXITED"


def test_wait_terminal_fails_after_bounded_missing_polls(monkeypatch):
    class Client:
        pass

    monkeypatch.setattr(launcher.storage, "maybe_read_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.storage, "pod_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "best_effort_log_snapshot", lambda *_: {"status": "captured", "bytes": 60, "tail": "pod not found"})
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    with pytest.raises(launcher.PodExitedWithoutTerminal) as exc:
        launcher.wait_terminal(
            Client(), object(), execution_id="exec", attempt=1, pod_id="pod-missing", silent_start_timeout_seconds=999999
        )
    assert exc.value.snapshot["desiredStatus"] == "MISSING"
    assert exc.value.snapshot["consecutive_missing_polls"] == 3


def test_wait_terminal_aborts_silent_running_pod(monkeypatch):
    class Client:
        pass

    monkeypatch.setattr(launcher.storage, "maybe_read_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.storage, "pod_snapshot", lambda *args, **kwargs: {"desiredStatus": "RUNNING", "id": "pod-silent"})
    monkeypatch.setattr(launcher, "best_effort_log_snapshot", lambda *_: {"status": "empty", "bytes": 0})
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    with pytest.raises(launcher.PodSilentStartup):
        launcher.wait_terminal(
            Client(), object(), execution_id="exec", attempt=1, pod_id="pod-silent", silent_start_timeout_seconds=0
        )


def test_terminal_verification_requires_no_cross_lane_ranking():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"cross_lane_ranking_performed": False' in source
    assert "scientific_diagnostic_scaling_recovery_model_complete" in source
