#!/usr/bin/env python3
"""Render or explicitly launch the bounded GLM cheap screen on RunPod."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_adaptation_elasticity_v1 as lifecycle
import launch_first_bounded_scientific_training as storage
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = ROOT / "configs/experiments/hephaestus_glm_cheap_screen_v1.json"
AUTH_ENV = "HEPHAESTUS_GLM_CHEAP_SCREEN_LAUNCH_AUTHORIZED"
TERMINAL_STATUSES = {"EXITED", "FAILED", "TERMINATED", "STOPPED"}


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def load_screen() -> dict[str, Any]:
    value = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("GLM cheap-screen protocol is not an object")
    return value


def pod_shell(screen: dict[str, Any]) -> str:
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
p=subprocess.run(
    ['nvidia-smi','--query-gpu=driver_version,name,memory.total,uuid,compute_cap','--format=csv,noheader'],
    capture_output=True,text=True,check=False,
)
d={'nvidia_smi_returncode':int(p.returncode),'nvidia_smi_stdout':p.stdout.strip(),'nvidia_smi_stderr':p.stderr.strip()}
print('GLM_CHEAP_SCREEN_GPU_JSON '+json.dumps(d,sort_keys=True),flush=True)
if p.returncode != 0 or 'RTX PRO 6000 Blackwell' not in p.stdout:
    raise SystemExit('GLM cheap-screen GPU preflight failed')
PYGPU
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv curl
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check \
  'torch==2.14.0+cu130' --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \
  'transformers==5.17.0' \
  'accelerate==1.15.0' \
  'safetensors==0.8.0' \
  'huggingface-hub==1.31.0'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" - <<'PYCUDA'
import json, os, torch
arches=list(torch.cuda.get_arch_list())
d={
    'torch':str(torch.__version__),
    'cuda':str(torch.version.cuda),
    'cuda_available':bool(torch.cuda.is_available()),
    'device_count':int(torch.cuda.device_count()),
    'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'device_capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    'arch_list':arches,
    'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
}
print('GLM_CHEAP_SCREEN_CUDA_JSON '+json.dumps(d,sort_keys=True),flush=True)
if not d['cuda_available'] or d['device_count'] != 1 or 'sm_120' not in arches:
    raise SystemExit('GLM cheap-screen Blackwell CUDA runtime preflight failed')
x=torch.ones(1,device='cuda'); torch.cuda.synchronize()
print('GLM_CHEAP_SCREEN_CUDA_OK '+str(float(x.item())),flush=True)
PYCUDA
"$PY" -m py_compile scripts/run_glm_cheap_screen_v1.py
"$PY" scripts/run_glm_cheap_screen_v1.py
'''


def placeholder_environment(repo_sha: str, run_id: str) -> dict[str, str]:
    return {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_GLM_SCREEN_RUN_ID": run_id,
        "RUNPOD_S3_ACCESS_KEY_ID": "<secret>",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "<secret>",
        "RUNPOD_S3_ENDPOINT_URL": "<configured-endpoint>",
        "RUNPOD_DATACENTER_ID": "<configured-region>",
        "RUNPOD_NETWORK_VOLUME_ID": "<s3-bucket-id-only-not-mounted>",
        "HF_HUB_DISABLE_XET": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
    }


def pod_environment(repo_sha: str, run_id: str) -> dict[str, str]:
    env = placeholder_environment(repo_sha, run_id)
    env.update({
        "RUNPOD_S3_ACCESS_KEY_ID": required("RUNPOD_S3_ACCESS_KEY_ID"),
        "RUNPOD_S3_SECRET_ACCESS_KEY": required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        "RUNPOD_S3_ENDPOINT_URL": required("RUNPOD_S3_ENDPOINT_URL"),
        "RUNPOD_DATACENTER_ID": required("RUNPOD_DATACENTER_ID"),
        "RUNPOD_NETWORK_VOLUME_ID": required("RUNPOD_NETWORK_VOLUME_ID"),
    })
    return env


def build_pod_request(screen: dict[str, Any], *, name: str, env: dict[str, str]) -> dict[str, Any]:
    execution = screen["execution"]
    body = {
        "name": name[:180],
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(execution["gpu_type_ids"]),
        "gpuTypePriority": "availability",
        "cloudType": execution["cloud_type"],
        "imageName": execution["image"],
        "containerDiskInGb": int(execution["container_disk_gb"]),
        "dockerStartCmd": ["bash", "-lc", pod_shell(screen)],
        "interruptible": False,
        "env": dict(env),
    }
    validate_pod_request(screen, body)
    return body


def validate_pod_request(screen: dict[str, Any], body: dict[str, Any]) -> None:
    execution = screen["execution"]
    if body.get("gpuCount") != 1:
        raise ValueError("cheap screen must use exactly one GPU")
    if body.get("gpuTypeIds") != execution["gpu_type_ids"]:
        raise ValueError("cheap-screen GPU allowlist drifted")
    if any("H200" in item or "B200" in item for item in body["gpuTypeIds"]):
        raise ValueError("expensive H200/B200 fallback is forbidden for cheap screen")
    if "networkVolumeId" in body or "volumeMountPath" in body:
        raise ValueError("cheap screen must remain ephemeral with no Network Volume attachment")
    if body.get("cloudType") != "SECURE":
        raise ValueError("cheap screen must remain on Secure Cloud")
    if body.get("imageName") != execution["image"]:
        raise ValueError("cheap-screen runtime image drifted")
    shell = str(body["dockerStartCmd"][-1])
    if "run_glm_cheap_screen_v1.py" not in shell:
        raise ValueError("cheap-screen Pod does not execute the screening runner")
    if "run_diagnostic_scaling_recovery_v1.py" in shell:
        raise ValueError("cheap-screen Pod must not execute full recovery")
    if "peft" in shell.casefold():
        raise ValueError("cheap-screen bootstrap unexpectedly installs PEFT/LoRA tooling")


def execute(screen: dict[str, Any]) -> dict[str, Any]:
    if str(os.environ.get(AUTH_ENV, "")).strip() != "YES":
        raise RuntimeError(f"paid cheap-screen launch requires {AUTH_ENV}=YES")
    governance = screen["governance"]
    if governance.get("paid_launch_allowed_now") is not True:
        raise RuntimeError("paid GLM cheap-screen launch is currently disabled by governance")

    repo_sha = required("GITHUB_SHA")
    github_run_id = required("GITHUB_RUN_ID")
    run_id = f"glm-cheap-screen-v1-{github_run_id}"
    prefix = f"{screen['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    result_key = f"{prefix}/result.json"
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = storage.s3_client()
    env = pod_environment(repo_sha, run_id)
    body = build_pod_request(screen, name=f"hephaestus-{run_id}", env=env)

    pod_id: str | None = None
    started = time.monotonic()
    record: dict[str, Any] = {
        "launcher_version": "hephaestus-glm-cheap-screen-launcher.v1",
        "run_id": run_id,
        "repo_sha": repo_sha,
        "result_key": result_key,
        "status": "starting",
        "gpu_type_ids": body["gpuTypeIds"],
        "max_hourly_usd": screen["execution"]["max_hourly_usd"],
        "max_estimated_total_usd": screen["execution"]["max_estimated_total_usd"],
        "hard_wall_seconds": screen["execution"]["hard_wall_seconds"],
    }
    try:
        pod = execution._create_pod(body)
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(screen["execution"]["hard_wall_seconds"]):
                raise TimeoutError("GLM cheap screen exceeded hard wall-clock cutoff")
            snap = storage.pod_snapshot(execution, pod_id)
            if snap is None:
                raise RuntimeError("GLM cheap-screen Pod disappeared before governed terminal result")
            raw_price = snap.get("costPerHr")
            if raw_price is not None:
                hourly = float(raw_price)
                estimated = hourly * elapsed / 3600.0
                record["cost_per_hr"] = hourly
                record["estimated_cost_usd"] = estimated
                if hourly > float(screen["execution"]["max_hourly_usd"]):
                    raise RuntimeError(f"cheap-screen hourly price ceiling exceeded: ${hourly:.4f}/hr")
                if estimated > float(screen["execution"]["max_estimated_total_usd"]):
                    raise RuntimeError(f"cheap-screen estimated total cost ceiling exceeded: ${estimated:.4f}")

            terminal = storage.maybe_read_key(client, result_key)
            if terminal is not None:
                result = json.loads(terminal)
                record["result"] = result
                record["status"] = "completed"
                print("GLM_CHEAP_SCREEN_LAUNCH_RESULT_JSON " + json.dumps(record, sort_keys=True), flush=True)
                return record

            status = str(snap.get("desiredStatus", "")).upper()
            if status in TERMINAL_STATUSES:
                raise RuntimeError(f"GLM cheap-screen Pod became terminal before S3 result: {status}")
            time.sleep(float(screen["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = lifecycle.delete_pod_with_retries(execution, pod_id)
            print("GLM_CHEAP_SCREEN_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("GLM cheap-screen Pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    screen = load_screen()
    repo_sha = os.environ.get("GITHUB_SHA", "<repo-sha>")
    run_id = f"glm-cheap-screen-v1-{os.environ.get('GITHUB_RUN_ID', '<github-run-id>')}"
    body = build_pod_request(screen, name=f"hephaestus-{run_id}", env=placeholder_environment(repo_sha, run_id))
    if not args.execute:
        print(json.dumps({
            "mode": "render_only",
            "protocol_id": screen["protocol_id"],
            "paid_launch_allowed_now": screen["governance"]["paid_launch_allowed_now"],
            "request": body,
            "limits": {
                "hard_wall_seconds": screen["execution"]["hard_wall_seconds"],
                "max_hourly_usd": screen["execution"]["max_hourly_usd"],
                "max_estimated_total_usd": screen["execution"]["max_estimated_total_usd"],
            },
        }, indent=2, sort_keys=True))
        return 0
    execute(screen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())