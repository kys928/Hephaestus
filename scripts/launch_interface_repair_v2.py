#!/usr/bin/env python3
"""Guarded launcher for Interface Repair V2 preflight or full single-pod stack."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter
from hephaestus.providers.runpod.execution import RunPodApiError

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v2.json"
MARKER_PATH = ROOT / "configs/experiments/interface_repair_v2.launch.json"
AUTH_ENV = "HEPHAESTUS_INTERFACE_REPAIR_V2_AUTHORIZED"
TERMINAL = {"EXITED", "FAILED", "TERMINATED", "STOPPED"}


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def authorize(cfg: dict[str, Any], marker: dict[str, Any], mode: str) -> None:
    if cfg["governance"].get("production_promotion_allowed") or cfg["governance"].get("automatic_role_dispatch_allowed"):
        raise RuntimeError("V2 cannot enable production promotion or automatic dispatch")
    if cfg.get("frozen_diagnosis") is not True:
        raise RuntimeError("Diagnosis must remain frozen")
    if mode == "preflight":
        allowed = cfg["governance"].get("paid_preflight_allowed")
        marked = marker.get("preflight_authorized")
        phrase = str(marker.get("authorization_phrase_preflight", ""))
    elif mode == "full":
        allowed = cfg["governance"].get("paid_full_launch_allowed")
        marked = marker.get("full_launch_authorized")
        phrase = str(marker.get("authorization_phrase_full", ""))
    else:
        raise ValueError(mode)
    if not allowed:
        raise RuntimeError(f"paid Interface Repair V2 {mode} disabled")
    if marked is not True:
        raise RuntimeError(f"Interface Repair V2 {mode} marker is not authorized")
    if not phrase or required(AUTH_ENV) != phrase:
        raise RuntimeError(f"Interface Repair V2 {mode} authorization phrase missing or mismatched")


def pod_shell(mode: str) -> str:
    if mode == "preflight":
        body = '''export HEPHAESTUS_REPAIR_ROLE=planner
"$PY" scripts/run_interface_repair_v2.py --preflight'''
    elif mode == "full":
        body = '''for ROLE in planner evaluator judge controller; do
  export HEPHAESTUS_REPAIR_ROLE="$ROLE"
  "$PY" scripts/run_interface_repair_v2.py
done
unset HEPHAESTUS_REPAIR_ROLE
"$PY" scripts/run_interface_repair_live_v2.py'''
    else:
        raise ValueError(mode)
    return rf'''set -Eeuo pipefail
export HF_HOME=/workspace/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/workspace/hephaestus-cache/xdg
export TMPDIR=/workspace/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
nvidia-smi --query-gpu=driver_version,name,memory.total,uuid,compute_cap --format=csv,noheader
if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
  rm -rf /var/lib/apt/lists/*
fi
rm -rf /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \
  'transformers==5.17.0' 'accelerate==1.15.0' 'safetensors==0.8.0' \
  'huggingface-hub==1.31.0' 'peft==0.21.0' 'mistral-common==1.11.7'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" -m py_compile scripts/run_interface_repair_v2.py scripts/run_interface_repair_live_v2.py scripts/launch_interface_repair_v2.py
"$PY" scripts/build_interface_repair_v1.py
{body}
'''


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, mode: str, *, real_env: bool) -> dict[str, object]:
    ex = cfg["execution"]
    env = {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_INTERFACE_REPAIR_RUN_ID": run_id,
        "RUNPOD_S3_ACCESS_KEY_ID": "<secret>",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "<secret>",
        "RUNPOD_S3_ENDPOINT_URL": "<endpoint>",
        "RUNPOD_DATACENTER_ID": "<region>",
        "RUNPOD_NETWORK_VOLUME_ID": "<volume>",
        "PYTHONUNBUFFERED": "1",
    }
    if real_env:
        for key in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY", "RUNPOD_S3_ENDPOINT_URL", "RUNPOD_DATACENTER_ID", "RUNPOD_NETWORK_VOLUME_ID"):
            env[key] = required(key)
    datacenter = required("RUNPOD_DATACENTER_ID") if real_env else "<region>"
    volume = required("RUNPOD_NETWORK_VOLUME_ID") if real_env else "<volume>"
    return {
        "name": f"hephaestus-interface-repair-v2-{mode}-{run_id}"[:180],
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(ex["gpu_type_ids"]),
        "gpuTypePriority": "custom",
        "cloudType": ex["cloud_type"],
        "dataCenterIds": [datacenter],
        "dataCenterPriority": "custom",
        "imageName": ex["image"],
        "containerDiskInGb": int(ex["container_disk_gb"]),
        "networkVolumeId": volume,
        "volumeMountPath": "/workspace",
        "dockerStartCmd": ["bash", "-lc", pod_shell(mode)],
        "interruptible": False,
        "env": env,
    }


def s3_client() -> Any:
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"), region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"), aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )


def maybe_json(client: Any, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"), Key=key)
    except Exception as exc:  # noqa: BLE001
        data = getattr(exc, "response", {})
        code = str(data.get("Error", {}).get("Code", "")) if isinstance(data, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        value = json.loads(response["Body"].read().decode("utf-8"))
    finally:
        response["Body"].close()
    return value if isinstance(value, dict) else None


def create_with_capacity_retries(execution: RunPodExecutionAdapter, body: dict[str, object], attempts: int, delay: float) -> dict[str, Any]:
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            pod = execution._create_pod(body)
            print("INTERFACE_REPAIR_V2_CAPACITY_JSON " + json.dumps({"attempt": attempt, "status": "created", "pod_id": pod.get("id")}, sort_keys=True), flush=True)
            return pod
        except RunPodApiError as exc:
            last = exc
            if "no instances currently available" not in str(exc).lower():
                raise
            print("INTERFACE_REPAIR_V2_CAPACITY_JSON " + json.dumps({"attempt": attempt, "status": "unavailable"}, sort_keys=True), flush=True)
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(f"A40 capacity unavailable after {attempts} attempts: {last}") from last


def delete_with_retries(execution: RunPodExecutionAdapter, pod_id: str) -> dict[str, object]:
    errors: list[str] = []
    for attempt in range(1, 6):
        try:
            execution.delete_pod(pod_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        time.sleep(min(attempt, 3))
        try:
            execution.get_pod(pod_id)
        except Exception:
            return {"deleted": True, "verified_absent": True, "attempts": attempt, "errors": errors}
    return {"deleted": False, "verified_absent": False, "attempts": 5, "errors": errors}


def execute(cfg: dict[str, Any], marker: dict[str, Any], mode: str) -> dict[str, Any]:
    authorize(cfg, marker, mode)
    repo_sha = required("GITHUB_SHA")
    github_run_id = required("GITHUB_RUN_ID")
    run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", f"interface-repair-v2-{mode}-{github_run_id}")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    if mode == "preflight":
        result_key = f"{prefix}/preflight/{cfg['preflight']['role']}/result.json"
        wall = int(cfg["execution"]["hard_wall_seconds_preflight"])
        max_total = float(cfg["execution"]["max_estimated_total_usd_preflight"])
    else:
        result_key = f"{prefix}/live/result.json"
        wall = int(cfg["execution"]["hard_wall_seconds_stack"])
        max_total = float(cfg["execution"]["max_estimated_total_usd_stack"])

    client = s3_client()
    client.head_bucket(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, mode, real_env=True)
    pod_id: str | None = None
    started: float | None = None
    record: dict[str, Any] = {"mode": mode, "run_id": run_id, "repo_sha": repo_sha, "status": "starting", "result_key": result_key}
    try:
        pod = create_with_capacity_retries(execution, body, int(cfg["execution"]["capacity_retry_attempts"]), float(cfg["execution"]["capacity_retry_seconds"]))
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        started = time.monotonic()
        print("INTERFACE_REPAIR_V2_POD_JSON " + json.dumps({"mode": mode, "pod_id": pod_id, "created_at_unix": time.time(), "gpu_allowlist": cfg["execution"]["gpu_type_ids"]}, sort_keys=True), flush=True)
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= wall:
                raise TimeoutError(f"V2 {mode} hard wall reached")
            snapshot = execution.get_pod(pod_id)
            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError(f"hourly cost ceiling exceeded: {hourly:.4f}")
                if estimated > max_total:
                    raise RuntimeError(f"estimated {mode} cost ceiling exceeded: {estimated:.4f}")
            terminal = maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                if terminal.get("status") == "completed":
                    if mode == "preflight" and terminal.get("approved_for_full_run") is not True:
                        record["status"] = "preflight_rejected"
                    else:
                        record["status"] = "completed"
                    return record
                if terminal.get("status") == "failed":
                    raise RuntimeError(f"V2 runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
            if str(snapshot.get("desiredStatus", "")).upper() in TERMINAL:
                raise RuntimeError("V2 pod became terminal before terminal S3 result appeared")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("INTERFACE_REPAIR_V2_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("V2 pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    mode = "preflight" if args.preflight else "full"
    cfg = load_json(CFG_PATH)
    marker = load_json(MARKER_PATH)
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>")
    run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", f"interface-repair-v2-{mode}-{os.environ.get('GITHUB_RUN_ID', '<run>')}")
    rendered = {
        "mode": mode,
        "run_id": run_id,
        "request": request_body(cfg, repo_sha, run_id, mode, real_env=False),
        "execution": cfg["execution"],
        "governance": cfg["governance"],
    }
    if not args.execute:
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, marker, mode)
    print("INTERFACE_REPAIR_V2_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
