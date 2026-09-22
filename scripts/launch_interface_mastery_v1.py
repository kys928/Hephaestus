#!/usr/bin/env python3
"""Render or execute the guarded Phase II Interface Mastery V1 RunPod launch.

Default mode is render-only and performs no network access.  Paid pod creation is
possible only with all three gates satisfied:
1. protocol governance permits a paid launch;
2. the frozen launch marker is explicitly authorized; and
3. the runtime authorization phrase is present in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_mastery_v1.json"
MARKER_PATH = ROOT / "configs/experiments/interface_mastery_v1.launch.json"
AUTH_ENV = "HEPHAESTUS_PHASE_II_LAUNCH_AUTHORIZED"
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


def frozen_authorization(cfg: dict[str, Any], marker: dict[str, Any]) -> None:
    phrase = str(marker.get("authorization_phrase", ""))
    if not bool(cfg["governance"].get("paid_launch_allowed")):
        raise RuntimeError("Phase II protocol does not permit a paid launch")
    if marker.get("authorized") is not True:
        raise RuntimeError("Phase II launch marker is not authorized")
    if not phrase or os.environ.get(AUTH_ENV, "").strip() != phrase:
        raise RuntimeError("Phase II runtime launch authorization phrase is missing or mismatched")


def pod_shell() -> str:
    return r'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
python - <<'PYGPU'
import json, subprocess
p=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name,memory.total,uuid,compute_cap','--format=csv,noheader'],capture_output=True,text=True)
print('INTERFACE_MASTERY_GPU_JSON '+json.dumps({'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()},sort_keys=True),flush=True)
if p.returncode != 0:
    raise SystemExit('nvidia-smi failed')
PYGPU
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
"$PY" -m py_compile scripts/build_interface_mastery_v1.py scripts/run_interface_mastery_v1.py
"$PY" scripts/build_interface_mastery_v1.py
"$PY" scripts/run_interface_mastery_v1.py
'''


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, *, real_env: bool) -> dict[str, object]:
    execution = cfg["execution"]
    env = {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_INTERFACE_MASTERY_RUN_ID": run_id,
        "RUNPOD_S3_ACCESS_KEY_ID": "<secret>",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "<secret>",
        "RUNPOD_S3_ENDPOINT_URL": "<endpoint>",
        "RUNPOD_DATACENTER_ID": "<region>",
        "RUNPOD_NETWORK_VOLUME_ID": "<volume>",
        "PYTHONUNBUFFERED": "1",
    }
    if real_env:
        for key in (
            "RUNPOD_S3_ACCESS_KEY_ID",
            "RUNPOD_S3_SECRET_ACCESS_KEY",
            "RUNPOD_S3_ENDPOINT_URL",
            "RUNPOD_DATACENTER_ID",
            "RUNPOD_NETWORK_VOLUME_ID",
        ):
            env[key] = required(key)
    return {
        "name": f"hephaestus-interface-mastery-{run_id}"[:180],
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(execution["gpu_type_ids"]),
        "gpuTypePriority": "custom",
        "cloudType": execution["cloud_type"],
        "imageName": execution["image"],
        "containerDiskInGb": int(execution["container_disk_gb"]),
        "dockerStartCmd": ["bash", "-lc", pod_shell()],
        "interruptible": False,
        "env": env,
    }


def s3_client() -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )


def maybe_result(client: Any, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"), Key=key)
    except Exception as exc:  # noqa: BLE001
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        payload = json.loads(response["Body"].read().decode("utf-8"))
    finally:
        response["Body"].close()
    if not isinstance(payload, dict):
        raise RuntimeError("Phase II result is not a JSON object")
    return payload


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


def execute(cfg: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    frozen_authorization(cfg, marker)
    repo_sha = required("GITHUB_SHA")
    github_run_id = required("GITHUB_RUN_ID")
    run_id = f"interface-mastery-v1-{github_run_id}"
    prefix = f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    result_key = f"{prefix}/result.json"

    client = s3_client()
    client.head_bucket(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, real_env=True)
    started = time.monotonic()
    pod_id: str | None = None
    record: dict[str, Any] = {
        "launcher_version": "interface-mastery-launcher.v1",
        "run_id": run_id,
        "repo_sha": repo_sha,
        "status": "starting",
        "result_key": result_key,
        "weights_mutation_allowed": False,
        "production_promotion_allowed": False,
    }
    try:
        pod = execution._create_pod(body)  # guarded external side-effect boundary
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(cfg["execution"]["hard_wall_seconds"]):
                raise TimeoutError("Phase II hard wall reached")
            snapshot = execution.get_pod(pod_id)
            cost_per_hr = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost_per_hr is not None:
                hourly = float(cost_per_hr)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd"]):
                    raise RuntimeError("Phase II hourly cost ceiling exceeded")
                if estimated > float(cfg["execution"]["max_estimated_total_usd"]):
                    raise RuntimeError("Phase II estimated total cost ceiling exceeded")
            terminal = maybe_result(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                if terminal.get("status") == "completed":
                    record["status"] = "completed"
                    return record
                if terminal.get("status") == "failed":
                    raise RuntimeError(f"Phase II runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
            if str(snapshot.get("desiredStatus", "")).upper() in TERMINAL:
                raise RuntimeError("Phase II pod became terminal before a terminal S3 result appeared")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("INTERFACE_MASTERY_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("Phase II pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CFG_PATH)
    marker = load_json(MARKER_PATH)
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>")
    run_id = f"interface-mastery-v1-{os.environ.get('GITHUB_RUN_ID', '<run>')}"
    rendered = {
        "mode": "execute" if args.execute else "render_only",
        "protocol_id": cfg["protocol_id"],
        "authorized": marker.get("authorized") is True,
        "authorization_phrase": marker.get("authorization_phrase"),
        "paid_launch_allowed": cfg["governance"].get("paid_launch_allowed") is True,
        "weights_mutation_allowed": cfg["governance"].get("weights_mutation_allowed") is True,
        "production_promotion_allowed": cfg["governance"].get("production_promotion_allowed") is True,
        "automatic_role_dispatch_allowed": cfg["governance"].get("automatic_role_dispatch_allowed") is True,
        "request": request_body(cfg, repo_sha, run_id, real_env=False),
        "limits": cfg["execution"],
    }
    if not args.execute:
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return 0
    record = execute(cfg, marker)
    print("INTERFACE_MASTERY_LAUNCH_RESULT_JSON " + json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())