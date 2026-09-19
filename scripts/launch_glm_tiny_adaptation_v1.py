#!/usr/bin/env python3
"""Render or explicitly launch the bounded GLM tiny-adaptation probe on RunPod."""
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
import run_glm_tiny_adaptation_v1 as tiny_runner
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/hephaestus_glm_tiny_adaptation_v1.json"
AUTH_ENV = "HEPHAESTUS_GLM_TINY_ADAPT_LAUNCH_AUTHORIZED"
TERMINAL_STATUSES = {"EXITED", "FAILED", "TERMINATED", "STOPPED"}


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("GLM tiny-adaptation protocol is not an object")
    return value


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
python - <<'PYCUDA'
import json, os, subprocess, torch
p=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name,memory.total,uuid','--format=csv,noheader'],capture_output=True,text=True,check=False)
d={'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'cuda_available':bool(torch.cuda.is_available()),'device_count':int(torch.cuda.device_count()),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'nvidia_smi_returncode':int(p.returncode),'nvidia_smi_stdout':p.stdout.strip(),'nvidia_smi_stderr':p.stderr.strip()}
print('GLM_TINY_ADAPT_CUDA_JSON '+json.dumps(d,sort_keys=True),flush=True)
if not d['cuda_available'] or d['device_count'] != 1: raise SystemExit('GLM tiny-adaptation CUDA preflight failed')
x=torch.ones(1,device='cuda'); torch.cuda.synchronize(); print('GLM_TINY_ADAPT_CUDA_OK '+str(float(x.item())),flush=True)
PYCUDA
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv curl
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \
  'transformers==5.17.0' \
  'accelerate==1.15.0' \
  'safetensors==0.8.0' \
  'huggingface-hub==1.31.0' \
  'peft==0.20.0'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" -m py_compile scripts/run_glm_tiny_adaptation_v1.py
"$PY" scripts/run_glm_tiny_adaptation_v1.py
'''


def placeholder_environment(repo_sha: str, run_id: str, screen_result_key: str) -> dict[str, str]:
    return {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_GLM_TINY_ADAPT_RUN_ID": run_id,
        "HEPHAESTUS_GLM_SCREEN_RESULT_KEY": screen_result_key,
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


def pod_environment(repo_sha: str, run_id: str, screen_result_key: str) -> dict[str, str]:
    env = placeholder_environment(repo_sha, run_id, screen_result_key)
    env.update({
        "RUNPOD_S3_ACCESS_KEY_ID": required("RUNPOD_S3_ACCESS_KEY_ID"),
        "RUNPOD_S3_SECRET_ACCESS_KEY": required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        "RUNPOD_S3_ENDPOINT_URL": required("RUNPOD_S3_ENDPOINT_URL"),
        "RUNPOD_DATACENTER_ID": required("RUNPOD_DATACENTER_ID"),
        "RUNPOD_NETWORK_VOLUME_ID": required("RUNPOD_NETWORK_VOLUME_ID"),
    })
    return env


def build_pod_request(protocol: dict[str, Any], *, name: str, env: dict[str, str]) -> dict[str, Any]:
    ex = protocol["execution"]
    body = {
        "name": name[:180], "computeType": "GPU", "gpuCount": 1,
        "gpuTypeIds": list(ex["gpu_type_ids"]), "gpuTypePriority": "availability",
        "cloudType": ex["cloud_type"], "imageName": ex["image"],
        "containerDiskInGb": int(ex["container_disk_gb"]),
        "dockerStartCmd": ["bash", "-lc", pod_shell()], "interruptible": False, "env": dict(env),
    }
    validate_pod_request(protocol, body)
    return body


def validate_pod_request(protocol: dict[str, Any], body: dict[str, Any]) -> None:
    ex=protocol["execution"]
    if body.get("gpuCount") != 1: raise ValueError("tiny adaptation must use exactly one GPU")
    if body.get("gpuTypeIds") != ex["gpu_type_ids"]: raise ValueError("tiny-adaptation GPU allowlist drifted")
    if any("H200" in x or "B200" in x for x in body["gpuTypeIds"]): raise ValueError("expensive fallback forbidden for tiny adaptation")
    if "networkVolumeId" in body or "volumeMountPath" in body: raise ValueError("tiny adaptation must remain ephemeral")
    if body.get("cloudType") != "SECURE": raise ValueError("tiny adaptation must remain Secure Cloud")
    shell=str(body["dockerStartCmd"][-1])
    if "run_glm_tiny_adaptation_v1.py" not in shell: raise ValueError("tiny-adaptation runner missing")
    if "run_diagnostic_scaling_recovery_v1.py" in shell: raise ValueError("tiny adaptation may not execute full recovery")
    if "peft==0.20.0" not in shell: raise ValueError("tiny adaptation must pin PEFT")


def execute(protocol: dict[str, Any]) -> dict[str, Any]:
    if str(os.environ.get(AUTH_ENV, "")).strip() != "YES":
        raise RuntimeError(f"paid tiny-adaptation launch requires {AUTH_ENV}=YES")
    if protocol["governance"].get("paid_launch_allowed_now") is not True:
        raise RuntimeError("paid GLM tiny-adaptation launch is currently disabled by governance")

    screen_result_key=required("HEPHAESTUS_GLM_SCREEN_RESULT_KEY")
    client=storage.s3_client()
    screen=json.loads(storage.read_key(client,screen_result_key))
    tiny_runner.validate_prerequisite(screen,protocol)

    repo_sha=required("GITHUB_SHA"); github_run_id=required("GITHUB_RUN_ID")
    run_id=f"glm-tiny-adaptation-v1-{github_run_id}"
    prefix=f"{protocol['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    result_key=f"{prefix}/result.json"
    execution=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider())
    env=pod_environment(repo_sha,run_id,screen_result_key)
    body=build_pod_request(protocol,name=f"hephaestus-{run_id}",env=env)
    pod_id=None; started=time.monotonic()
    record={"launcher_version":"hephaestus-glm-tiny-adaptation-launcher.v1","run_id":run_id,"repo_sha":repo_sha,"result_key":result_key,"screen_result_key":screen_result_key,"status":"starting","gpu_type_ids":body["gpuTypeIds"],"max_hourly_usd":protocol["execution"]["max_hourly_usd"],"max_estimated_total_usd":protocol["execution"]["max_estimated_total_usd"],"hard_wall_seconds":protocol["execution"]["hard_wall_seconds"]}
    try:
        pod=execution._create_pod(body); pod_id=str(pod["id"]); record["pod_id"]=pod_id
        while True:
            elapsed=time.monotonic()-started
            if elapsed >= float(protocol["execution"]["hard_wall_seconds"]): raise TimeoutError("GLM tiny adaptation exceeded hard wall-clock cutoff")
            snap=storage.pod_snapshot(execution,pod_id)
            if snap is None: raise RuntimeError("GLM tiny-adaptation Pod disappeared before terminal result")
            if snap.get("costPerHr") is not None:
                hourly=float(snap["costPerHr"]); estimated=hourly*elapsed/3600.0
                record["cost_per_hr"]=hourly; record["estimated_cost_usd"]=estimated
                if hourly>float(protocol["execution"]["max_hourly_usd"]): raise RuntimeError(f"tiny-adaptation hourly price ceiling exceeded: ${hourly:.4f}/hr")
                if estimated>float(protocol["execution"]["max_estimated_total_usd"]): raise RuntimeError(f"tiny-adaptation estimated total cost ceiling exceeded: ${estimated:.4f}")
            terminal=storage.maybe_read_key(client,result_key)
            if terminal is not None:
                result=json.loads(terminal); record["result"]=result; record["status"]="completed"
                print("GLM_TINY_ADAPTATION_LAUNCH_RESULT_JSON "+json.dumps(record,sort_keys=True),flush=True); return record
            status=str(snap.get("desiredStatus","")).upper()
            if status in TERMINAL_STATUSES: raise RuntimeError(f"GLM tiny-adaptation Pod became terminal before S3 result: {status}")
            time.sleep(float(protocol["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown=lifecycle.delete_pod_with_retries(execution,pod_id)
            print("GLM_TINY_ADAPTATION_TEARDOWN_JSON "+json.dumps(teardown,sort_keys=True),flush=True)
            if not teardown.get("verified_absent"): raise RuntimeError("GLM tiny-adaptation Pod teardown could not be verified")


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); parser.add_argument("--screen-result-key",default="<screen-result-key>"); args=parser.parse_args()
    protocol=load_protocol(); repo_sha=os.environ.get("GITHUB_SHA","<repo-sha>"); run_id=f"glm-tiny-adaptation-v1-{os.environ.get('GITHUB_RUN_ID','<github-run-id>')}"
    screen_key=os.environ.get("HEPHAESTUS_GLM_SCREEN_RESULT_KEY",args.screen_result_key)
    body=build_pod_request(protocol,name=f"hephaestus-{run_id}",env=placeholder_environment(repo_sha,run_id,screen_key))
    if not args.execute:
        print(json.dumps({"mode":"render_only","protocol_id":protocol["protocol_id"],"paid_launch_allowed_now":protocol["governance"]["paid_launch_allowed_now"],"request":body,"limits":{"hard_wall_seconds":protocol["execution"]["hard_wall_seconds"],"max_hourly_usd":protocol["execution"]["max_hourly_usd"],"max_estimated_total_usd":protocol["execution"]["max_estimated_total_usd"]}},indent=2,sort_keys=True)); return 0
    execute(protocol); return 0
if __name__=="__main__": raise SystemExit(main())