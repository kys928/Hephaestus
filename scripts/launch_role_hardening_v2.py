#!/usr/bin/env python3
"""Launch and monitor one Phase II role-hardening training pod."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_adaptation_elasticity_v1 as lifecycle
import launch_first_bounded_scientific_training as storage
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/experiments/hephaestus_role_hardening_v2.json"
MARKER = ROOT / "configs/experiments/role_hardening_v2.launch.json"
AUTH = "HEPHAESTUS_ROLE_HARDENING_V2_AUTHORIZED"
TERMINAL = {"EXITED", "FAILED", "TERMINATED", "STOPPED"}


def req(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError("missing required environment variable: " + name)
    return value


def load_cfg() -> dict:
    return json.loads(CFG.read_text())


def load_marker() -> dict:
    return json.loads(MARKER.read_text())


def shell(role: str) -> str:
    return f'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
nvidia-smi
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
"$PY" scripts/build_role_hardening_v2.py
"$PY" -m py_compile scripts/run_role_hardening_v2.py
"$PY" scripts/run_role_hardening_v2.py --role {role}
'''


def env(repo_sha: str, run_id: str, role: str) -> dict[str, str]:
    return {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_ROLE_HARDENING_RUN_ID": run_id,
        "HEPHAESTUS_ROLE": role,
        "RUNPOD_S3_ACCESS_KEY_ID": req("RUNPOD_S3_ACCESS_KEY_ID"),
        "RUNPOD_S3_SECRET_ACCESS_KEY": req("RUNPOD_S3_SECRET_ACCESS_KEY"),
        "RUNPOD_S3_ENDPOINT_URL": req("RUNPOD_S3_ENDPOINT_URL"),
        "RUNPOD_DATACENTER_ID": req("RUNPOD_DATACENTER_ID"),
        "RUNPOD_NETWORK_VOLUME_ID": req("RUNPOD_NETWORK_VOLUME_ID"),
        "PYTHONUNBUFFERED": "1",
    }


def body(cfg: dict, role: str, repo_sha: str, run_id: str) -> dict:
    execution = cfg["execution"]
    return {
        "name": f"hephaestus-role-hardening-v2-{role}-{run_id}"[:180],
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": execution["gpu_type_ids"],
        "gpuTypePriority": "availability",
        "cloudType": execution["cloud_type"],
        "imageName": execution["image"],
        "containerDiskInGb": execution["container_disk_gb"],
        "dockerStartCmd": ["bash", "-lc", shell(role)],
        "interruptible": False,
        "env": env(repo_sha, run_id, role),
    }


def maybe(client, key):
    for i in range(5):
        try:
            return storage.maybe_read_key(client, key)
        except Exception as exc:
            if type(exc).__name__ != "FlexibleChecksumError":
                raise
            time.sleep(min(2**i, 8))
    return None


def execute(role: str) -> dict:
    cfg = load_cfg()
    marker = load_marker()
    if role not in cfg["candidates"]:
        raise RuntimeError("unknown role " + role)
    if os.environ.get(AUTH) != "YES" or marker.get("authorized") is not True:
        raise RuntimeError("role hardening launch is not explicitly authorized")
    if marker.get("pack_sha256") != cfg["pack"]["canonical_sha256"]:
        raise RuntimeError("hardening launch marker pack hash mismatch")
    if marker.get("frozen_redteam_pack_sha256") != cfg["source_stack"]["original_redteam_pack_sha256"]:
        raise RuntimeError("hardening launch marker red-team hash mismatch")
    if cfg["governance"]["automatic_role_dispatch_enable_allowed"] is not False:
        raise RuntimeError("hardening protocol must not enable automatic role dispatch")

    repo_sha = req("GITHUB_SHA")
    gid = req("GITHUB_RUN_ID")
    run_id = f"role-hardening-v2-{gid}"
    prefix = f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}/roles/{role}"
    result_key = f"{prefix}/result.json"
    progress_key = f"{prefix}/progress.json"
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = storage.s3_client()
    pod_id = None
    started = time.monotonic()
    record = {"role": role, "run_id": run_id, "repo_sha": repo_sha, "status": "starting"}

    try:
        observations: list[dict[str, object]] = []
        pod = lifecycle.retry_transient(
            lambda: execution._create_pod(body(cfg, role, repo_sha, run_id)),
            label=f"create_role_hardening_v2_{role}",
            observations=observations,
        )
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        record["create_observations"] = observations
        print("ROLE_HARDENING_V2_POD_JSON " + json.dumps(record, sort_keys=True), flush=True)

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(cfg["execution"]["hard_wall_seconds"]):
                raise TimeoutError("role hardening V2 hard wall reached")
            snapshot = storage.pod_snapshot(execution, pod_id)
            if snapshot is None:
                raise RuntimeError("pod disappeared before terminal result")
            if snapshot.get("costPerHr") is not None:
                hourly = float(snapshot["costPerHr"])
                cost = hourly * elapsed / 3600
                record.update(cost_per_hr=hourly, estimated_cost_usd=cost)
                if hourly > float(cfg["execution"]["max_hourly_usd"]):
                    raise RuntimeError("hourly cost ceiling exceeded")
                if cost > float(cfg["execution"]["max_estimated_usd_per_role"]):
                    raise RuntimeError("per-role cost ceiling exceeded")

            raw = maybe(client, progress_key)
            if raw:
                progress = json.loads(raw)
                record["last_progress"] = progress
                print(
                    "ROLE_HARDENING_V2_MONITOR_JSON "
                    + json.dumps(
                        {
                            "role": role,
                            "pod_id": pod_id,
                            "cost_per_hr": record.get("cost_per_hr"),
                            "estimated_cost_usd": record.get("estimated_cost_usd"),
                            "progress": progress,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            terminal = maybe(client, result_key)
            if terminal:
                result = json.loads(terminal)
                if result.get("repo_sha") != repo_sha:
                    raise RuntimeError("hardening result repo SHA mismatch")
                if result.get("status") == "failed":
                    raise RuntimeError(
                        f"hardening runner failed: {result.get('error_type')}: {result.get('error')}"
                    )
                if result.get("status") == "completed":
                    record["status"] = "completed"
                    record["result"] = result
                    print(
                        "ROLE_HARDENING_V2_LAUNCH_RESULT_JSON "
                        + json.dumps(record, sort_keys=True),
                        flush=True,
                    )
                    return record

            if str(snapshot.get("desiredStatus", "")).upper() in TERMINAL:
                raise RuntimeError("pod terminal before S3 result")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = lifecycle.delete_pod_with_retries(execution, pod_id)
            print(
                "ROLE_HARDENING_V2_TEARDOWN_JSON "
                + json.dumps({"role": role, **teardown}, sort_keys=True),
                flush=True,
            )
            if not teardown.get("verified_absent"):
                raise RuntimeError("role hardening V2 pod teardown unverified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        cfg = load_cfg()
        print(
            json.dumps(
                {
                    "role": args.role,
                    "authorized": load_marker().get("authorized"),
                    "candidate": cfg["candidates"].get(args.role),
                    "execution": cfg["execution"],
                },
                indent=2,
            )
        )
        return
    execute(args.role)


if __name__ == "__main__":
    main()
