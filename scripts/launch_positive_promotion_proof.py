#!/usr/bin/env python3
"""Launch and independently verify the real-model positive promotion proof."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

import launch_first_bounded_scientific_training as base
from runpod_capacity_selection import create_with_capacity_retries

VOLUME_ID = base.VOLUME_ID
DATACENTER_ID = base.DATACENTER_ID
SCIENTIFIC_PREFIX = base.SCIENTIFIC_PREFIX
IMAGE = base.IMAGE
EVAL_PACK_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"
APPROVAL_REF = "approval://operator/chat-2026-09-05-positive-real-compute-promotion"
MAX_ATTEMPTS = 3
MAX_SECONDS = 2400
POLL_SECONDS = 5


def atomic_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def volume_key(path: str) -> str:
    if not path.startswith("/workspace/"):
        raise RuntimeError(f"proof path is not Network-Volume local: {path}")
    return path[len("/workspace/") :]


def is_funds_error(message: str) -> bool:
    lowered = message.lower()
    return "http 402" in lowered or any(
        phrase in lowered
        for phrase in (
            "insufficient funds",
            "insufficient credit",
            "insufficient balance",
            "not enough credit",
        )
    )


def pod_shell() -> str:
    return r'''set -Eeuo pipefail
ATTEMPT_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_PROOF_RUN_ID}/attempt-${HEPHAESTUS_ATTEMPT}"
mkdir -p "$ATTEMPT_DIR"
exec >"$ATTEMPT_DIR/pod_runtime.log" 2>&1
write_bootstrap_failure() {
  code=$?
  if [ "$code" -ne 0 ] && [ ! -f "$ATTEMPT_DIR/driver_result.json" ]; then
    python - "$ATTEMPT_DIR/driver_result.json" "$code" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
payload = {
    "result_version": "positive-real-model-promotion-proof.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "proof_run_id": os.environ.get("HEPHAESTUS_PROOF_RUN_ID", "unknown"),
    "attempt": os.environ.get("HEPHAESTUS_ATTEMPT", "unknown"),
    "status": "pod_bootstrap_failed",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
    "training_performed": False,
    "original_research_lineage_mutated": False,
}
with open(path + ".partial", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
os.replace(path + ".partial", path)
PYFAIL
  fi
}
trap write_bootstrap_failure EXIT
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --disable-pip-version-check -e . 'transformers>=4.46,<6' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1' 'huggingface_hub>=0.26,<2' 'hf_xet>=1,<2'
"$PY" - <<'PYCHECK'
import torch
assert torch.cuda.is_available(), "CUDA unavailable after positive-proof bootstrap"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PYCHECK
"$PY" -m py_compile \
  src/hephaestus/control/semantic_judge.py \
  src/hephaestus/production/actions.py \
  src/hephaestus/production/loop.py \
  scripts/run_positive_promotion_proof.py
"$PY" scripts/run_positive_promotion_proof.py
'''


def create_pod(
    execution: RunPodExecutionAdapter,
    *,
    proof_run_id: str,
    repo_sha: str,
    attempt: int,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    def create_once(gpu_ids: list[str]) -> dict[str, Any]:
        body: dict[str, object] = {
            "name": f"hephaestus-positive-promotion-{proof_run_id}-a{attempt}"[:180],
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": list(gpu_ids),
            "gpuTypePriority": "availability",
            "cloudType": "SECURE",
            "dataCenterIds": [DATACENTER_ID],
            "dataCenterPriority": "custom",
            "imageName": IMAGE,
            "containerDiskInGb": 24,
            "networkVolumeId": VOLUME_ID,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", pod_shell()],
            "interruptible": False,
            "env": {
                "HEPHAESTUS_PROOF_RUN_ID": proof_run_id,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_ATTEMPT": str(attempt),
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": APPROVAL_REF,
            },
        }
        return execution._create_pod(body)  # integration-owned scheduler projection

    return create_with_capacity_retries(create_once)


def wait_for_result(
    client: Any,
    execution: RunPodExecutionAdapter,
    *,
    proof_run_id: str,
    attempt: int,
    pod_id: str,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    key = f"{SCIENTIFIC_PREFIX}/executions/{proof_run_id}/attempt-{attempt}/driver_result.json"
    deadline = time.monotonic() + MAX_SECONDS
    observations: list[dict[str, object]] = []
    last_status: str | None = None
    while time.monotonic() < deadline:
        raw = base.maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("positive-proof terminal record is not an object")
            return payload, observations
        pod = base.pod_snapshot(execution, pod_id)
        status = str(pod.get("desiredStatus", "unknown")) if pod else "unknown"
        if status != last_status:
            observations.append({"at": datetime.now(timezone.utc).isoformat(), "desired_status": status})
            last_status = status
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"positive promotion proof did not finish within {MAX_SECONDS} seconds")


def verify_proof(client: Any, proof_run_id: str, result: dict[str, Any]) -> dict[str, object]:
    if result.get("status") != "completed":
        raise RuntimeError(f"positive proof did not complete: {result.get('status')}: {result.get('error', '')}")
    if result.get("training_performed") is not False:
        raise RuntimeError("positive promotion proof unexpectedly performed training")
    if result.get("original_research_lineage_mutated") is not False:
        raise RuntimeError("positive promotion proof reports original research lineage mutation")

    verification_path = str(result.get("independent_verification_ref") or "")
    expected_verification_hash = str(result.get("independent_verification_sha256") or "")
    raw_verification = base.read_key(client, volume_key(verification_path))
    observed_hash = f"sha256:{hashlib.sha256(raw_verification).hexdigest()}"
    if observed_hash != expected_verification_hash:
        raise RuntimeError("independent verification S3 hash disagrees with terminal result")
    verification = json.loads(raw_verification.decode("utf-8"))
    if not isinstance(verification, dict) or verification.get("status") != "verified":
        raise RuntimeError("independent verification record is not verified")
    if verification.get("frozen_eval_pack_hash") != EVAL_PACK_HASH:
        raise RuntimeError("positive proof frozen eval-pack identity drifted")
    if verification.get("operator_approval_ref") != APPROVAL_REF:
        raise RuntimeError("positive proof operator approval reference drifted")

    lineage = verification.get("lineage")
    manifest = verification.get("certified_model_manifest")
    if not isinstance(lineage, dict) or not isinstance(manifest, dict):
        raise RuntimeError("positive proof verification lacks lineage/model manifest")
    certified = str(verification.get("certified_checkpoint_ref") or "")
    if not certified:
        raise RuntimeError("positive proof has no certified checkpoint")
    if lineage.get("certified_stable_checkpoint_ref") != certified:
        raise RuntimeError("lineage certified checkpoint differs from verification")
    if lineage.get("best_checkpoint_ref") != certified or lineage.get("last_stable_checkpoint_ref") != certified:
        raise RuntimeError("lineage best/stable/certified refs disagree")
    if lineage.get("last_certification_result") != "certification_passed":
        raise RuntimeError("lineage certification state is not passed")
    if manifest.get("status") != "verified" or manifest.get("revision") not in {
        "c89bee90d9f811437d9735454613c35b4a3c4dc8",
        "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
    }:
        raise RuntimeError("certified model immutable revision is unexpected")
    if str(manifest.get("license") or "").lower() != "apache-2.0":
        raise RuntimeError("certified model is not Apache-2.0")

    proof_result_key = f"{SCIENTIFIC_PREFIX}/positive_promotion/{proof_run_id}/proof_result.json"
    proof_raw = base.read_key(client, proof_result_key)
    proof_payload = json.loads(proof_raw.decode("utf-8"))
    if not isinstance(proof_payload, dict) or proof_payload.get("certified_checkpoint_ref") != certified:
        raise RuntimeError("proof-result readback differs from independent verification")

    return {
        "verification_version": "positive-real-model-promotion-launcher-verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "proof_run_id": proof_run_id,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "proof_result": {
            "key": proof_result_key,
            "sha256": f"sha256:{hashlib.sha256(proof_raw).hexdigest()}",
            "bytes": len(proof_raw),
        },
        "independent_verification": {
            "key": volume_key(verification_path),
            "sha256": observed_hash,
            "bytes": len(raw_verification),
        },
        "certified_checkpoint_ref": certified,
        "certified_model_manifest": manifest,
        "lineage": lineage,
        "program_state": verification.get("program_state"),
        "terminal_result": result,
    }


def main() -> int:
    base.required("RUNPOD_API_KEY")
    repo_sha = base.required("GITHUB_SHA")
    workflow_run = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
    proof_run_id = f"positive-real-model-promotion-001-{workflow_run}"
    client = base.s3_client()
    client.head_bucket(Bucket=VOLUME_ID)
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())

    attempt_rows: list[dict[str, object]] = []
    launcher_error: str | None = None
    funds_unavailable = False
    final_result: dict[str, Any] | None = None
    verification: dict[str, object] | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        pod: dict[str, Any] | None = None
        pod_id: str | None = None
        row: dict[str, object] = {
            "attempt": attempt,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pod_id": None,
            "capacity_selection": [],
            "observations": [],
            "teardown": {"deleted": False, "not_created": True},
        }
        try:
            pod, capacity = create_pod(
                execution,
                proof_run_id=proof_run_id,
                repo_sha=repo_sha,
                attempt=attempt,
            )
            row["capacity_selection"] = capacity
            pod_id = str(pod["id"])
            row["pod_id"] = pod_id
            row["teardown"] = {"deleted": False, "not_created": False}
            result, observations = wait_for_result(
                client,
                execution,
                proof_run_id=proof_run_id,
                attempt=attempt,
                pod_id=pod_id,
            )
            row["observations"] = observations
            row["terminal_status"] = result.get("status")
            row["terminal_error"] = result.get("error")
            final_result = result
            if result.get("status") == "completed":
                verification = verify_proof(client, proof_run_id, result)
                row["verified"] = True
                attempt_rows.append(row)
                break
            row["verified"] = False
            launcher_error = f"proof attempt {attempt} failed: {result.get('status')}: {result.get('error', '')}"
        except BaseException as exc:
            launcher_error = f"{type(exc).__name__}: {exc}"
            row["launcher_error"] = launcher_error
            funds_unavailable = is_funds_error(launcher_error)
            if funds_unavailable:
                row["funds_unavailable"] = True
        finally:
            if pod_id:
                row["teardown"] = base.delete_pod(execution, pod_id)
            row["completed_at"] = datetime.now(timezone.utc).isoformat()
            if not attempt_rows or attempt_rows[-1] is not row:
                attempt_rows.append(row)
        if funds_unavailable:
            break

    launcher = {
        "launcher_version": "positive-real-model-promotion-launcher.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_sha": repo_sha,
        "workflow_run_id": workflow_run,
        "proof_run_id": proof_run_id,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "attempts": attempt_rows,
        "funds_unavailable": funds_unavailable,
        "error": launcher_error,
        "status": "verified" if verification else "failed",
        "terminal_result": final_result,
    }
    atomic_json(Path("positive_promotion_launcher.json"), launcher)
    if verification is not None:
        atomic_json(Path("positive_promotion_verification.json"), verification)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "proof_run_id": proof_run_id,
                    "certified_model": verification["certified_model_manifest"]["model_id"],
                    "certified_revision": verification["certified_model_manifest"]["revision"],
                    "attempt_count": len(attempt_rows),
                    "funds_unavailable": False,
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(launcher, indent=2, sort_keys=True))
    return 2 if funds_unavailable else 1


if __name__ == "__main__":
    raise SystemExit(main())
