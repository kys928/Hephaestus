#!/usr/bin/env python3
"""Guarded RunPod launcher for one interface-repair role or the live stack certification."""
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
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v1.json"
MARKER_PATH = ROOT / "configs/experiments/interface_repair_v1.launch.json"
AUTH_ENV = "HEPHAESTUS_INTERFACE_REPAIR_LAUNCH_AUTHORIZED"
TERMINAL = {"EXITED", "FAILED", "TERMINATED", "STOPPED"}


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def authorization(cfg: dict[str, Any], marker: dict[str, Any]) -> None:
    phrase = str(marker.get("authorization_phrase", ""))
    if not cfg["governance"].get("paid_launch_allowed"):
        raise RuntimeError("paid interface-repair launch disabled")
    if marker.get("authorized") is not True:
        raise RuntimeError("interface-repair launch marker is not authorized")
    if not phrase or os.environ.get(AUTH_ENV, "").strip() != phrase:
        raise RuntimeError("interface-repair authorization phrase missing or mismatched")
    if cfg["governance"].get("production_promotion_allowed") or cfg["governance"].get("automatic_role_dispatch_allowed"):
        raise RuntimeError("repair experiment cannot enable production promotion or automatic dispatch")
    if not cfg["governance"].get("frozen_diagnosis", cfg.get("frozen_diagnosis", False)) and cfg.get("frozen_diagnosis") is not True:
        raise RuntimeError("Diagnosis must remain frozen")


def pod_shell(mode: str, role: str | None) -> str:
    if mode == "role":
        command = '"$PY" scripts/run_interface_repair_v1.py'
    elif mode == "live":
        command = '"$PY" scripts/run_interface_repair_live_v1.py'
    else:
        raise ValueError(mode)
    role_export = f"export HEPHAESTUS_REPAIR_ROLE={role!s}\n" if role else ""
    return rf'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
{role_export}mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
nvidia-smi --query-gpu=driver_version,name,memory.total,uuid,compute_cap --format=csv,noheader
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \
  'transformers==5.17.0' 'accelerate==1.15.0' 'safetensors==0.8.0' \
  'huggingface-hub==1.31.0' 'peft==0.21.0' 'mistral-common==1.11.7'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" -m py_compile scripts/build_interface_repair_v1.py scripts/run_interface_repair_v1.py
"$PY" scripts/build_interface_repair_v1.py
{command}
'''


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, mode: str, role: str | None, *, real_env: bool) -> dict[str, object]:
    execution = cfg["execution"]
    env = {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_INTERFACE_REPAIR_RUN_ID": run_id,
        "RUNPOD_S3_ACCESS_KEY_ID": "<secret>", "RUNPOD_S3_SECRET_ACCESS_KEY": "<secret>",
        "RUNPOD_S3_ENDPOINT_URL": "<endpoint>", "RUNPOD_DATACENTER_ID": "<region>", "RUNPOD_NETWORK_VOLUME_ID": "<volume>",
        "PYTHONUNBUFFERED": "1",
    }
    if role:
        env["HEPHAESTUS_REPAIR_ROLE"] = role
    if real_env:
        for key in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY", "RUNPOD_S3_ENDPOINT_URL", "RUNPOD_DATACENTER_ID", "RUNPOD_NETWORK_VOLUME_ID"):
            env[key] = required(key)
    return {
        "name": f"hephaestus-interface-repair-{mode}-{role or 'stack'}-{run_id}"[:180],
        "computeType": "GPU", "gpuCount": 1, "gpuTypeIds": list(execution["gpu_type_ids"]), "gpuTypePriority": "custom",
        "cloudType": execution["cloud_type"], "imageName": execution["image"], "containerDiskInGb": int(execution["container_disk_gb"]),
        "dockerStartCmd": ["bash", "-lc", pod_shell(mode, role)], "interruptible": False, "env": env,
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
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        value = json.loads(response["Body"].read().decode("utf-8"))
    finally:
        response["Body"].close()
    if not isinstance(value, dict):
        raise RuntimeError(f"{key} is not a JSON object")
    return value


def create_with_capacity_retries(
    execution: RunPodExecutionAdapter,
    body: dict[str, object],
    *,
    attempts: int,
    delay_seconds: float,
) -> dict[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            pod = execution._create_pod(body)
            print("INTERFACE_REPAIR_CAPACITY_JSON " + json.dumps({"attempt": attempt, "status": "created", "pod_id": pod.get("id")}, sort_keys=True), flush=True)
            return pod
        except RunPodApiError as exc:
            last_error = exc
            lowered = str(exc).lower()
            if "no instances currently available" not in lowered:
                raise
            print("INTERFACE_REPAIR_CAPACITY_JSON " + json.dumps({"attempt": attempt, "status": "unavailable", "error": str(exc)}, sort_keys=True), flush=True)
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(f"RunPod capacity unavailable after {attempts} attempts: {last_error}") from last_error


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


def execute(cfg: dict[str, Any], marker: dict[str, Any], mode: str, role: str | None) -> dict[str, Any]:
    authorization(cfg, marker)
    if mode == "role" and role not in cfg["trainable_roles"]:
        raise RuntimeError(f"invalid repair role: {role}")
    repo_sha = required("GITHUB_SHA"); github_run_id = required("GITHUB_RUN_ID")
    run_id = required("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID") if os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID") else f"interface-repair-v1-{github_run_id}"
    prefix = f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    result_key = f"{prefix}/roles/{role}/result.json" if mode == "role" else f"{prefix}/live/result.json"
    wall = int(cfg["execution"]["hard_wall_seconds_per_role"] if mode == "role" else cfg["execution"]["hard_wall_seconds_live"])
    max_total = float(cfg["execution"]["max_estimated_total_usd_per_training_role"] if mode == "role" else cfg["execution"]["max_estimated_total_usd_live"])

    client = s3_client(); client.head_bucket(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, mode, role, real_env=True)
    pod_id: str | None = None; started: float | None = None
    record: dict[str, Any] = {"mode": mode, "role": role, "run_id": run_id, "repo_sha": repo_sha, "status": "starting", "result_key": result_key}
    try:
        pod = create_with_capacity_retries(
            execution,
            body,
            attempts=int(cfg["execution"].get("capacity_retry_attempts", 1)),
            delay_seconds=float(cfg["execution"].get("capacity_retry_seconds", 10)),
        )
        pod_id = str(pod["id"]); record["pod_id"] = pod_id; started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= wall:
                raise TimeoutError(f"{mode} hard wall reached")
            snapshot = execution.get_pod(pod_id)
            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost); estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                hourly_limit = float(cfg["execution"]["max_hourly_usd_per_pod"])
                if hourly > hourly_limit:
                    raise RuntimeError(f"hourly cost ceiling exceeded: {hourly:.4f} > {hourly_limit:.4f}")
                if estimated > max_total:
                    raise RuntimeError(f"estimated total cost ceiling exceeded: {estimated:.4f} > {max_total:.4f}")
            terminal = maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                if terminal.get("status") == "completed":
                    record["status"] = "completed"; return record
                if terminal.get("status") == "failed":
                    raise RuntimeError(f"runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
            if str(snapshot.get("desiredStatus", "")).upper() in TERMINAL:
                raise RuntimeError("pod became terminal before terminal S3 result appeared")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = delete_with_retries(execution, pod_id); record["teardown"] = teardown
            print("INTERFACE_REPAIR_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--role"); parser.add_argument("--live", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); cfg = load_json(CFG_PATH); marker = load_json(MARKER_PATH)
    mode = "live" if args.live else "role"; role = None if args.live else args.role
    if mode == "role" and not role:
        raise RuntimeError("--role is required unless --live is used")
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>"); run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", f"interface-repair-v1-{os.environ.get('GITHUB_RUN_ID', '<run>')}")
    rendered = {"mode": mode, "role": role, "authorized": marker.get("authorized") is True, "run_id": run_id,
                "request": request_body(cfg, repo_sha, run_id, mode, role, real_env=False), "limits": cfg["execution"]}
    if not args.execute:
        print(json.dumps(rendered, indent=2, sort_keys=True)); return 0
    result = execute(cfg, marker, mode, role)
    print("INTERFACE_REPAIR_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
