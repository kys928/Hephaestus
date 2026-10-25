#!/usr/bin/env python3
"""One operator-authorized, bounded RunPod five-role CUDA smoke and cleanup.

Launches exactly ONE new disposable Secure Cloud Pod in the pinned datacenter.
No model promotion or dataset/evaluation modification. Verifies a terminal
volume receipt by authenticated S3 GET and always attempts Pod termination.
"""
from __future__ import annotations

import json
import os
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

import preflight_cuda_five_role_campaign as billing
import preflight_five_role_s3_artifacts as custody

MAX_POD_SECONDS = 1180
MAX_RATE_PER_HOUR = 2.65  # UI-verified 96 GB RTX PRO 6000 WK ceiling
MIN_CURRENT_CREDITS_USD = 6.0
RUN_RESULT = Path("evidence/cuda_five_role_launcher_result.json")


def _save(doc: dict) -> None:
    RUN_RESULT.parent.mkdir(parents=True, exist_ok=True)
    RUN_RESULT.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")


def _shell() -> str:
    return r'''set -Eeuo pipefail
ROOT="/workspace/hephaestus/research/cuda-roles-20261025/$HEPHAESTUS_ROLE_RUN_ID"
mkdir -p "$ROOT"
exec > "$ROOT/pod_runtime.log" 2>&1
write_failure() {
  local code="$?"
  if [ "$code" -ne 0 ] && [ ! -f "$ROOT/role_result.json" ]; then
    printf '{"schema":"hephaestus-real-five-role-cuda-inference.v1","status":"pod_bootstrap_failed","full_45_stage_corpus_campaign":false,"scientific_certificate_issued":false}\n' > "$ROOT/role_result.json"
  fi
}
trap write_failure EXIT
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends git ca-certificates
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /tmp/hephaestus-role-src
cd /tmp/hephaestus-role-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m pip install --disable-pip-version-check -e . \
  'transformers>=5.0,<6' 'peft>=0.16,<1' \
  'huggingface-hub>=0.35,<2' 'mistral-common>=1.8,<3' \
  'safetensors>=0.4,<1' 'accelerate>=1.0,<3'
python scripts/run_frozen_five_role_cuda_probe.py
'''


def _s3():
    return boto3.client(
        "s3", endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(connect_timeout=10, read_timeout=30,
                      retries={"max_attempts": 3, "mode": "standard"}))


def main() -> int:
    result = {
        "schema": "hephaestus-cuda-runpod-bounded-launch.v1",
        "status": "blocked",
        "pod_created": False, "pod_deleted": False,
        "production_activation": False,
        "certification": "not_certified",
        "evidence": {},
    }
    pod_id = None
    _save(result)
    try:
        if billing.main() != 0 or custody.main() != 0:
            raise PermissionError("RunPod custody/balance preflight failed")
        before = json.loads(Path(os.environ.get(
            "HEPHAESTUS_CUDA_PREFLIGHT_OUTPUT", "evidence/cuda_preflight.json")).read_text())
        assets = json.loads(Path(os.environ.get(
            "HEPHAESTUS_ROLE_ASSET_REPORT", "evidence/cuda_role_assets.json")).read_text())
        if (not assets.get("complete")
                or type(before.get("current_balance_usd")) not in (float, int)
                or before["current_balance_usd"] < MIN_CURRENT_CREDITS_USD):
            raise PermissionError("credits or real frozen role adapters not adequate")
        if not all(item.get("verified") for item in assets.get("role_assets", [])):
            raise PermissionError("unverified frozen S3 adapter")
        result["balance_at_start_usd"] = before["current_balance_usd"]
        result["verified_adapters"] = len(assets["role_assets"])
        secrets = EnvironmentSecretsProvider()
        config = RunPodConfig.from_env()
        execution = RunPodExecutionAdapter(
            config=config, secrets=secrets,
            state_root=Path("evidence/runpod-role-operation-state"),
        )
        # Never use a pre-existing Pod belonging to a different project.
        existing_ids = {item["id"] for item in execution.list_pods()}
        allowed = [
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        ]
        selected = execution.select_gpu(allowed)
        result["gpu_id"] = selected
        result["gpu_hourly_price_limit_usd"] = MAX_RATE_PER_HOUR
        # Catalog availability alone does not substantiate a price. Restrict
        # to operator-observed EU-CZ-1 hardware class instead of fallback.
        if selected not in allowed:
            raise PermissionError("catalog chose unapproved GPU")
        run_id = "hephaestus-five-role-cuda-" + os.environ["GITHUB_RUN_ID"]
        sha = os.environ["GITHUB_SHA"]
        if len(sha) != 40 or not all(ch in "0123456789abcdef" for ch in sha):
            raise PermissionError("exact Git repository SHA is mandatory")
        pod = execution.create_bounded_gpu_pod(
            name=run_id,
            image_name="pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime",
            gpu_type_ids=[selected],
            docker_start_cmd=["bash", "-lc", _shell()],
            env={
                "HEPHAESTUS_ROLE_RUN_ID": run_id,
                "HEPHAESTUS_REPO_SHA": sha,
                "HEPHAESTUS_HF_CACHE_DIR": "/tmp/hephaestus-hf-cache",
                "HEPHAESTUS_ROLE_RUN_ROOT":
                    "/workspace/hephaestus/research/cuda-roles-20261025/" + run_id,
            },
            container_disk_in_gb=95,
            operation_id="operator-approved-" + run_id,
        )
        pod_id = str(pod["id"])
        if not pod_id or pod_id in existing_ids:
            raise PermissionError("new Pod identity collides with existing resource")
        result.update(status="pod_running", pod_created=True, pod_id=pod_id)
        _save(result)
        source = _s3()
        target = ("hephaestus/research/cuda-roles-20261025/" +
                  run_id + "/role_result.json")
        deadline = time.monotonic() + MAX_POD_SECONDS
        while time.monotonic() < deadline:
            try:
                data = source.get_object(
                    Bucket=config.network_volume_id, Key=target)["Body"].read(4_000_000)
                doc = json.loads(data)
                if doc.get("status") in {
                    "five_frozen_role_inferences_completed",
                    "partial_failure", "failed_before_model_execution",
                    "pod_bootstrap_failed",
                }:
                    result["evidence"] = {
                        "volume_key": target, "stage_status": doc.get("status"),
                        "role_result_count": len(doc.get("role_results", [])),
                        "raw_receipt_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                    }
                    Path("evidence/cuda_role_output.json").write_bytes(data)
                    result["status"] = (
                        "model_execution_completed" if
                        doc.get("status") == "five_frozen_role_inferences_completed"
                        else "model_execution_failed")
                    break
            except Exception:
                pass
            time.sleep(12)
        else:
            result["status"] = "bounded_deadline_exceeded"
        _save(result)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:600]
        if not result["pod_created"]:
            result["status"] = "blocked_before_pod_create"
        else:
            result["status"] = "launcher_error"
    finally:
        if pod_id is None and result.get("status") == "launcher_error":
            # Recover a Pod created during an ambiguous API reply only by its
            # exact new run name; leave every unrelated pre-existing Pod alone.
            try:
                expected = "hephaestus-five-role-cuda-" + os.environ["GITHUB_RUN_ID"]
                found = [
                    item for item in execution.list_pods()
                    if item.get("name") == expected and item.get("id") not in existing_ids
                ]
                if len(found) == 1:
                    pod_id = found[0]["id"]
                    result["reconciled_pod_id"] = pod_id
            except Exception as exc:
                result["reconciliation_error"] = type(exc).__name__
        if pod_id is not None:
            # Refuse to leak a paid Pod on expected paths.
            try:
                execution.delete_pod(pod_id)
                result["pod_deleted"] = True
            except Exception as exc:
                result["cleanup_error_type"] = type(exc).__name__
                result["pod_deleted"] = False
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        _save(result)
    print(json.dumps({key: result.get(key) for key in (
        "schema", "status", "pod_created", "pod_deleted", "gpu_id",
        "balance_at_start_usd", "evidence", "error_type", "error_message",
    )}, sort_keys=True))
    return 0 if result["status"] == "model_execution_completed" and result["pod_deleted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
