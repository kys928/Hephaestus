#!/usr/bin/env python3
"""Guarded launcher for Interface Repair V2 preflight or full single-pod stack."""
from __future__ import annotations

import argparse
import json
import os
import shlex
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
# Keep the historical constant name because tests and evidence tooling already consume it.
PRELIGHT_BENCHMARK_STAGES = {
    "preflight_evaluation_started",
    "preflight_eval_progress",
    "preflight_training_started",
    "training_progress",
    "preflight_complete",
}


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
        if not marker.get("preflight_evidence_key") or not marker.get("preflight_repo_sha"):
            raise RuntimeError("full V2 launch requires persisted preflight evidence bound to a repository SHA")
    else:
        raise ValueError(mode)
    if not allowed:
        raise RuntimeError(f"paid Interface Repair V2 {mode} disabled")
    if marked is not True:
        raise RuntimeError(f"Interface Repair V2 {mode} marker is not authorized")
    if not phrase or required(AUTH_ENV) != phrase:
        raise RuntimeError(f"Interface Repair V2 {mode} authorization phrase missing or mismatched")


def pod_shell(mode: str, venv_cache: str, pip_cache: str, adapter_cache: str) -> str:
    if mode == "preflight":
        body = '''export HEPHAESTUS_REPAIR_ROLE=planner
"$PY" scripts/run_interface_repair_v2_cached.py --preflight'''
    elif mode == "full":
        body = '''for ROLE in planner evaluator judge controller; do
  export HEPHAESTUS_REPAIR_ROLE="$ROLE"
  "$PY" scripts/run_interface_repair_v2_cached.py
done
unset HEPHAESTUS_REPAIR_ROLE
"$PY" scripts/run_interface_repair_live_v2.py'''
    else:
        raise ValueError(mode)

    qvenv = shlex.quote(venv_cache)
    qpip = shlex.quote(pip_cache)
    qadapter = shlex.quote(adapter_cache)
    return rf'''set -Eeuo pipefail
export HF_HOME=/workspace/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/workspace/hephaestus-cache/xdg
export TMPDIR=/workspace/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PIP_CACHE_DIR={qpip}
export HEPHAESTUS_V2_ADAPTER_CACHE={qadapter}
VENV={qvenv}
READY="$VENV/.hephaestus-ready"
BOOTSTRAP_FILE="/workspace/$HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$HEPHAESTUS_V2_ADAPTER_CACHE" "$(dirname "$VENV")" "$(dirname "$BOOTSTRAP_FILE")"

write_bootstrap() {{
  STAGE="$1" python - <<'PY'
import json, os, pathlib, subprocess, time
path = pathlib.Path(os.environ['BOOTSTRAP_FILE'])
payload = {{
    'stage': os.environ['STAGE'],
    'timestamp_unix': time.time(),
    'venv_ready': pathlib.Path(os.environ['READY']).is_file(),
    'hf_cache_root_exists': pathlib.Path(os.environ['HUGGINGFACE_HUB_CACHE']).exists(),
}}
try:
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,uuid,driver_version', '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.STDOUT, timeout=15,
    ).strip()
    if raw:
        payload['nvidia_smi'] = raw
except Exception as exc:
    payload['nvidia_smi_error'] = type(exc).__name__
tmp = path.with_suffix(path.suffix + '.tmp')
tmp.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
os.replace(tmp, path)
print('INTERFACE_REPAIR_V2_BOOTSTRAP_JSON ' + json.dumps(payload, sort_keys=True), flush=True)
PY
}}
export BOOTSTRAP_FILE READY HUGGINGFACE_HUB_CACHE

write_bootstrap container_started
if ! command -v git >/dev/null 2>&1 || ! python -m venv --help >/dev/null 2>&1; then
  write_bootstrap base_packages_install_started
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
  rm -rf /var/lib/apt/lists/*
fi

write_bootstrap repo_checkout_started
rm -rf /opt/hephaestus-src
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
write_bootstrap repo_checkout_complete

write_bootstrap environment_check_started
if [ ! -x "$VENV/bin/python" ]; then
  # Do not delete a partial cache. venv creation is idempotent enough to repair the
  # directory, and pip below resumes from the persistent wheel/download cache.
  python -m venv --system-site-packages "$VENV"
fi
PY="$VENV/bin/python"

ENV_OK=0
if "$PY" - <<'PY' >/dev/null 2>&1
import importlib.metadata as m
expected = {{
    'transformers': '5.17.0',
    'accelerate': '1.15.0',
    'safetensors': '0.8.0',
    'huggingface-hub': '1.31.0',
    'peft': '0.21.0',
    'mistral-common': '1.11.7',
}}
for package, version in expected.items():
    if m.version(package) != version:
        raise SystemExit(1)
import boto3, transformers, accelerate, safetensors, peft, mistral_common  # noqa: F401
PY
then
  ENV_OK=1
fi

if [ "$ENV_OK" -ne 1 ]; then
  write_bootstrap environment_install_started
  "$PY" -m pip install --disable-pip-version-check \
    'boto3>=1.35,<2' \
    'transformers==5.17.0' 'accelerate==1.15.0' 'safetensors==0.8.0' \
    'huggingface-hub==1.31.0' 'peft==0.21.0' 'mistral-common==1.11.7'
  "$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
  "$PY" - <<'PY'
import importlib.metadata as m
expected = {{
    'transformers': '5.17.0',
    'accelerate': '1.15.0',
    'safetensors': '0.8.0',
    'huggingface-hub': '1.31.0',
    'peft': '0.21.0',
    'mistral-common': '1.11.7',
}}
for package, version in expected.items():
    observed = m.version(package)
    if observed != version:
        raise RuntimeError(f'{{package}}={{observed}} expected {{version}}')
import boto3, transformers, accelerate, safetensors, peft, mistral_common  # noqa: F401
PY
  touch "$READY"
fi
write_bootstrap environment_ready

# Refresh only this repository's editable metadata. Heavy dependencies remain in
# the persistent venv and are never re-downloaded merely because the source SHA changed.
"$PY" -m pip install --disable-pip-version-check --no-deps -e . >/dev/null
"$PY" -m py_compile \
  scripts/run_interface_repair_v2.py \
  scripts/run_interface_repair_v2_cached.py \
  scripts/interface_repair_v2_bootstrap.py \
  scripts/run_interface_repair_live_v2.py \
  scripts/launch_interface_repair_v2.py
write_bootstrap pack_build_started
"$PY" scripts/build_interface_repair_v1.py
write_bootstrap runner_started
{body}
'''


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, mode: str, *, real_env: bool) -> dict[str, object]:
    ex = cfg["execution"]
    bootstrap_key = f"{str(ex['s3_prefix']).rstrip('/')}/{run_id}/bootstrap/progress.json"
    env = {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_INTERFACE_REPAIR_RUN_ID": run_id,
        "HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY": bootstrap_key,
        "RUNPOD_S3_ACCESS_KEY_ID": "<secret>",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "<secret>",
        "RUNPOD_S3_ENDPOINT_URL": "<endpoint>",
        "RUNPOD_DATACENTER_ID": "<region>",
        "RUNPOD_NETWORK_VOLUME_ID": "<volume>",
        "PYTHONUNBUFFERED": "1",
    }
    if real_env:
        for key in (
            "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY", "RUNPOD_S3_ENDPOINT_URL",
            "RUNPOD_DATACENTER_ID", "RUNPOD_NETWORK_VOLUME_ID",
        ):
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
        "dockerStartCmd": [
            "bash", "-lc",
            pod_shell(
                mode,
                str(ex["persistent_venv_cache"]),
                str(ex["persistent_pip_cache"]),
                str(ex["persistent_adapter_cache"]),
            ),
        ],
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


def verify_preflight_evidence(client: Any, marker: dict[str, Any], repo_sha: str) -> dict[str, Any]:
    key = str(marker.get("preflight_evidence_key") or "")
    bound_sha = str(marker.get("preflight_repo_sha") or "")
    if not key or not bound_sha:
        raise RuntimeError("missing preflight evidence binding")
    if bound_sha != repo_sha:
        raise RuntimeError(f"preflight repository SHA mismatch: {bound_sha} != {repo_sha}")
    result = maybe_json(client, key)
    if result is None:
        raise RuntimeError("bound preflight evidence does not exist")
    if result.get("status") != "completed" or result.get("approved_for_full_run") is not True:
        raise RuntimeError("bound preflight did not approve a full run")
    if str(result.get("repo_sha", "")) != repo_sha:
        raise RuntimeError("preflight result is not for the exact full-run repository SHA")
    return result


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
    raise RuntimeError(f"compatible GPU capacity unavailable after {attempts} attempts: {last}") from last


def safe_pod_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in (
        "id", "name", "desiredStatus", "createdAt", "lastStartedAt",
        "costPerHr", "adjustedCostPerHr", "gpuTypeId", "gpuType", "gpuDisplayName", "gpuName",
        "memoryInGb", "vcpuCount", "dataCenterId",
    ):
        value = snapshot.get(key)
        if isinstance(value, (str, int, float, bool)) and value is not None:
            safe[key] = value
    for parent_key in ("gpu", "machine"):
        parent = snapshot.get(parent_key)
        if isinstance(parent, dict):
            filtered = {
                key: value
                for key, value in parent.items()
                if key in {"id", "name", "displayName", "memoryInGb", "memoryInGB", "securePrice", "communityPrice", "dataCenterId", "gpuTypeId", "gpuDisplayName"}
                and isinstance(value, (str, int, float, bool))
            }
            if filtered:
                safe[parent_key] = filtered
    return safe


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
    bootstrap_key = f"{prefix}/bootstrap/progress.json"
    if mode == "preflight":
        role = str(cfg["preflight"]["role"])
        result_key = f"{prefix}/preflight/{role}/result.json"
        progress_key = f"{prefix}/preflight/{role}/progress.json"
        wall = int(cfg["execution"]["hard_wall_seconds_preflight_total"])
        bootstrap_wall = int(cfg["execution"]["hard_wall_seconds_preflight_bootstrap"])
        benchmark_wall = int(cfg["execution"]["hard_wall_seconds_preflight_benchmark"])
        max_total = float(cfg["execution"]["max_estimated_total_usd_preflight"])
    else:
        result_key = f"{prefix}/live/result.json"
        progress_key = None
        wall = int(cfg["execution"]["hard_wall_seconds_stack"])
        bootstrap_wall = benchmark_wall = 0
        max_total = float(cfg["execution"]["max_estimated_total_usd_stack"])

    client = s3_client()
    client.head_bucket(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"))
    preflight_evidence = verify_preflight_evidence(client, marker, repo_sha) if mode == "full" else None
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, mode, real_env=True)
    pod_id: str | None = None
    started: float | None = None
    benchmark_started: float | None = None
    last_progress_stage: str | None = None
    last_bootstrap_stage: str | None = None
    metadata_printed = False
    record: dict[str, Any] = {
        "mode": mode,
        "run_id": run_id,
        "repo_sha": repo_sha,
        "status": "starting",
        "result_key": result_key,
        "bootstrap_progress_key": bootstrap_key,
        "preflight_evidence_key": marker.get("preflight_evidence_key") if mode == "full" else None,
        "preflight_projection_seconds": preflight_evidence.get("projected_full_role_seconds") if preflight_evidence else None,
    }
    try:
        pod = create_with_capacity_retries(
            execution,
            body,
            int(cfg["execution"]["capacity_retry_attempts"]),
            float(cfg["execution"]["capacity_retry_seconds"]),
        )
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        started = time.monotonic()
        print("INTERFACE_REPAIR_V2_POD_JSON " + json.dumps({
            "mode": mode,
            "pod_id": pod_id,
            "created_at_unix": time.time(),
            "gpu_allowlist": cfg["execution"]["gpu_type_ids"],
        }, sort_keys=True), flush=True)

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= wall:
                raise TimeoutError(f"V2 {mode} total hard wall reached")

            snapshot = execution.get_pod(pod_id)
            metadata = safe_pod_metadata(snapshot)
            if metadata and not metadata_printed:
                print("INTERFACE_REPAIR_V2_POD_METADATA_JSON " + json.dumps(metadata, sort_keys=True), flush=True)
                record["pod_metadata"] = metadata
                metadata_printed = True

            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError(f"hourly cost ceiling exceeded: {hourly:.4f}")
                if estimated > max_total:
                    raise RuntimeError(f"estimated {mode} cost ceiling exceeded: {estimated:.4f}")

            bootstrap = maybe_json(client, bootstrap_key)
            if bootstrap is not None:
                stage = str(bootstrap.get("stage", ""))
                if stage and stage != last_bootstrap_stage:
                    print("INTERFACE_REPAIR_V2_BOOTSTRAP_PROGRESS_JSON " + json.dumps(bootstrap, sort_keys=True), flush=True)
                    record["last_bootstrap"] = bootstrap
                    last_bootstrap_stage = stage

            if mode == "preflight" and progress_key is not None:
                progress = maybe_json(client, progress_key)
                if progress is not None:
                    stage = str(progress.get("stage", ""))
                    if stage and stage != last_progress_stage:
                        print("INTERFACE_REPAIR_V2_PROGRESS_JSON " + json.dumps(progress, sort_keys=True), flush=True)
                        record["last_progress"] = progress
                        last_progress_stage = stage
                    if benchmark_started is None and stage in PRELIGHT_BENCHMARK_STAGES:
                        benchmark_started = time.monotonic()
                        record["benchmark_started_at_unix"] = time.time()
                        print("INTERFACE_REPAIR_V2_BENCHMARK_CLOCK_JSON " + json.dumps({
                            "status": "started",
                            "stage": stage,
                            "benchmark_wall_seconds": benchmark_wall,
                        }, sort_keys=True), flush=True)
                if benchmark_started is None and elapsed >= bootstrap_wall:
                    raise TimeoutError(
                        "V2 preflight bootstrap/model-init hard wall reached before benchmark start; "
                        f"last_bootstrap_stage={last_bootstrap_stage!r} last_runner_stage={last_progress_stage!r}"
                    )
                if benchmark_started is not None and (time.monotonic() - benchmark_started) >= benchmark_wall:
                    raise TimeoutError("V2 preflight benchmark hard wall reached")

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
        "preflight_evidence_key": marker.get("preflight_evidence_key"),
        "preflight_repo_sha": marker.get("preflight_repo_sha"),
    }
    if not args.execute:
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, marker, mode)
    print("INTERFACE_REPAIR_V2_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    if result.get("status") == "preflight_rejected":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
