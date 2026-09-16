#!/usr/bin/env python3
"""Render or explicitly launch one ephemeral Diagnostic Scaling V1 RunPod.

Default behavior is render-only and cannot create a Pod. Actual creation requires
both --execute and HEPHAESTUS_DIAGNOSTIC_SCALING_LAUNCH_AUTHORIZED=YES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
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
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json"
MAX_SECONDS = 20500
POLL_SECONDS = 15
CREATE_ATTEMPTS = 12
CREATE_RETRY_SECONDS = 10
AUTH_ENV = "HEPHAESTUS_DIAGNOSTIC_SCALING_LAUNCH_AUTHORIZED"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "--" for ch in value).strip("-.")


def load_contract() -> dict[str, Any]:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Diagnostic Scaling contract is not an object")
    return value


def select_candidate(contract: dict[str, Any], model_id: str) -> dict[str, Any]:
    candidate = next((row for row in contract["candidates"] if row["model_id"] == model_id), None)
    if candidate is None:
        raise ValueError(f"model is not in frozen Diagnostic Scaling cohort: {model_id}")
    return candidate


def gpu_ids(contract: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    if int(candidate["minimum_gpu_memory_gib"]) >= 140:
        return list(contract["execution"]["preferred_gpu_for_30b"])
    return list(contract["execution"]["preferred_gpu_for_14b"])


def pod_shell(contract: dict[str, Any]) -> str:
    deps = contract["execution"]["runtime_dependencies"]
    return f'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \\
  'transformers=={deps['transformers']}' \\
  'accelerate=={deps['accelerate']}' \\
  'safetensors=={deps['safetensors']}' \\
  'huggingface-hub=={deps['huggingface_hub']}' \\
  'peft=={deps['peft']}'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" -m py_compile scripts/run_diagnostic_scaling_v1.py
"$PY" scripts/run_diagnostic_scaling_v1.py
'''


def pod_environment(*, repo_sha: str, run_id: str, execution_id: str, attempt: int, model_id: str) -> dict[str, str]:
    return {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_DS_RUN_ID": run_id,
        "HEPHAESTUS_EXECUTION_ID": execution_id,
        "HEPHAESTUS_ATTEMPT": str(attempt),
        "HEPHAESTUS_MODEL_ID": model_id,
        "RUNPOD_S3_ACCESS_KEY_ID": required("RUNPOD_S3_ACCESS_KEY_ID"),
        "RUNPOD_S3_SECRET_ACCESS_KEY": required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        "RUNPOD_S3_ENDPOINT_URL": required("RUNPOD_S3_ENDPOINT_URL"),
        "RUNPOD_DATACENTER_ID": required("RUNPOD_DATACENTER_ID"),
        "RUNPOD_NETWORK_VOLUME_ID": required("RUNPOD_NETWORK_VOLUME_ID"),
        "HF_HUB_DISABLE_XET": "1",
        "PYTHONUNBUFFERED": "1",
    }


def placeholder_environment(*, repo_sha: str, run_id: str, execution_id: str, attempt: int, model_id: str) -> dict[str, str]:
    return {
        "HEPHAESTUS_REPO_SHA": repo_sha,
        "HEPHAESTUS_DS_RUN_ID": run_id,
        "HEPHAESTUS_EXECUTION_ID": execution_id,
        "HEPHAESTUS_ATTEMPT": str(attempt),
        "HEPHAESTUS_MODEL_ID": model_id,
        "RUNPOD_S3_ACCESS_KEY_ID": "<secret>",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "<secret>",
        "RUNPOD_S3_ENDPOINT_URL": "<configured-endpoint>",
        "RUNPOD_DATACENTER_ID": "EU-CZ-1",
        "RUNPOD_NETWORK_VOLUME_ID": "<s3-bucket-id-only-not-mounted>",
        "HF_HUB_DISABLE_XET": "1",
        "PYTHONUNBUFFERED": "1",
    }


def build_pod_request(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    *,
    name: str,
    env: dict[str, str],
) -> dict[str, object]:
    execution = contract["execution"]
    body: dict[str, object] = {
        "name": name[:180],
        "computeType": "GPU",
        "gpuCount": int(execution["gpu_count_per_pod"]),
        "gpuTypeIds": gpu_ids(contract, candidate),
        "gpuTypePriority": "availability",
        "cloudType": execution["cloud_type"],
        "dataCenterIds": [execution["datacenter_id"]],
        "dataCenterPriority": "custom",
        "imageName": execution["image"],
        "containerDiskInGb": int(execution["container_disk_gb"]),
        "dockerStartCmd": ["bash", "-lc", pod_shell(contract)],
        "interruptible": False,
        "env": dict(env),
    }
    validate_pod_request(contract, candidate, body)
    return body


def validate_pod_request(contract: dict[str, Any], candidate: dict[str, Any], body: dict[str, object]) -> None:
    execution = contract["execution"]
    if "networkVolumeId" in body or "volumeMountPath" in body:
        raise ValueError("ephemeral Diagnostic Scaling Pod request must not attach a Network Volume")
    if execution.get("network_volume_attached") is not False:
        raise ValueError("contract no longer requires ephemeral execution")
    if body.get("containerDiskInGb") != execution["container_disk_gb"]:
        raise ValueError("Pod container disk differs from frozen contract")
    if body.get("imageName") != execution["image"]:
        raise ValueError("Pod image differs from frozen contract")
    if body.get("gpuTypeIds") != gpu_ids(contract, candidate):
        raise ValueError("Pod GPU allowlist differs from candidate contract")
    if body.get("gpuCount") != 1 or body.get("cloudType") != "SECURE":
        raise ValueError("Pod compute topology differs from frozen contract")
    if body.get("dataCenterIds") != [execution["datacenter_id"]]:
        raise ValueError("Pod datacenter differs from frozen contract")
    shell = str((body.get("dockerStartCmd") or ["", "", ""])[-1])
    if "git checkout --detach \"$HEPHAESTUS_REPO_SHA\"" not in shell:
        raise ValueError("Pod bootstrap does not checkout the immutable admitted repository SHA")
    if "scripts/run_diagnostic_scaling_v1.py" not in shell:
        raise ValueError("Pod bootstrap does not invoke the frozen Diagnostic Scaling driver")


def redacted_request(body: dict[str, object]) -> dict[str, object]:
    copy = json.loads(json.dumps(body))
    env = copy.get("env")
    if isinstance(env, dict):
        for key in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"):
            if key in env:
                env[key] = "<redacted>"
    return copy


def admission_key(repo_sha: str) -> str:
    return f"{storage.SCIENTIFIC_PREFIX}/diagnostic_scaling/model_admission/{repo_sha}/admission.json"


def verify_admission(client: Any, *, repo_sha: str, contract: dict[str, Any], model_id: str) -> dict[str, Any]:
    raw = storage.read_key(client, admission_key(repo_sha))
    admission = json.loads(raw)
    if admission.get("status") != "launch_inputs_admitted" or admission.get("repo_sha") != repo_sha:
        raise RuntimeError("Diagnostic Scaling launch admission is missing or mismatched")
    if admission.get("protocol_sha256") != sha(CONTRACT_PATH.read_bytes()):
        raise RuntimeError("Diagnostic Scaling launch admission protocol hash drifted")
    candidate = select_candidate(contract, model_id)
    if admission.get("candidate_revisions", {}).get(model_id) != candidate["revision"]:
        raise RuntimeError("Diagnostic Scaling candidate revision was not admitted")
    if admission.get("network_volume_attached") is not False or admission.get("promotion_allowed") is not False:
        raise RuntimeError("Diagnostic Scaling launch admission violates execution/governance boundary")
    return admission


def next_attempt(client: Any, execution_id: str) -> int:
    prefix = f"{storage.SCIENTIFIC_PREFIX}/executions/{execution_id}/attempt-"
    attempts: set[int] = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=storage.VOLUME_ID, Prefix=prefix):
        for item in page.get("Contents", []):
            suffix = str(item.get("Key", ""))[len(prefix):]
            head = suffix.split("/", 1)[0]
            try:
                attempts.add(int(head))
            except ValueError:
                pass
    return max(attempts, default=0) + 1


def maybe_existing(client: Any, *, run_id: str, model_id: str, contract: dict[str, Any]) -> dict[str, Any] | None:
    key = f"{storage.SCIENTIFIC_PREFIX}/diagnostic_scaling/{run_id}/models/{slug(model_id)}/evidence_manifest.json"
    raw = storage.maybe_read_key(client, key)
    if raw is None:
        return None
    manifest = json.loads(raw)
    candidate = select_candidate(contract, model_id)
    if manifest.get("model_id") != model_id or manifest.get("revision") != candidate["revision"]:
        raise RuntimeError("existing Diagnostic Scaling evidence identity mismatch")
    if manifest.get("protocol_sha256") != sha(CONTRACT_PATH.read_bytes()):
        raise RuntimeError("existing Diagnostic Scaling evidence protocol mismatch")
    return {"key": key, "sha256": sha(raw), "bytes": len(raw), "manifest": manifest}


def wait_terminal(
    client: Any,
    execution: RunPodExecutionAdapter,
    *,
    execution_id: str,
    attempt: int,
    pod_id: str,
) -> dict[str, Any]:
    key = f"{storage.SCIENTIFIC_PREFIX}/executions/{execution_id}/attempt-{attempt}/driver_result.json"
    deadline = time.monotonic() + MAX_SECONDS
    while time.monotonic() < deadline:
        raw = storage.maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("Diagnostic Scaling terminal record is not an object")
            return payload
        storage.pod_snapshot(execution, pod_id)
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Diagnostic Scaling model did not finish within {MAX_SECONDS} seconds")


def verify_terminal(client: Any, result: dict[str, Any], *, run_id: str, execution_id: str, model_id: str, contract: dict[str, Any]) -> dict[str, Any]:
    candidate = select_candidate(contract, model_id)
    if result.get("status") != "completed" or result.get("disposition") != "scientific_diagnostic_scaling_model_complete":
        raise RuntimeError(f"Diagnostic Scaling model did not complete: {result.get('status')}: {result.get('disposition')}")
    expected = {
        "run_id": run_id,
        "execution_id": execution_id,
        "model_id": model_id,
        "revision": candidate["revision"],
        "protocol_sha256": sha(CONTRACT_PATH.read_bytes()),
        "network_volume_attached": False,
        "training_performed": True,
        "promotion_performed": False,
        "lineage_mutated": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Diagnostic Scaling terminal invariant mismatch: {key}")
    if set(result.get("roles", {})) != {"diagnosis", "controller"}:
        raise RuntimeError("Diagnostic Scaling terminal role coverage is incomplete")
    manifest = result.get("evidence_manifest")
    if not isinstance(manifest, dict) or not manifest.get("key") or not manifest.get("sha256"):
        raise RuntimeError("Diagnostic Scaling terminal record lacks evidence manifest identity")
    raw = storage.read_key(client, str(manifest["key"]))
    if sha(raw) != str(manifest["sha256"]):
        raise RuntimeError("Diagnostic Scaling evidence manifest failed independent S3 hash verification")
    return {
        "verification_version": "diagnostic-scaling-launcher-verification.v1",
        "verified_at": now(),
        "run_id": run_id,
        "execution_id": execution_id,
        "model_id": model_id,
        "revision": candidate["revision"],
        "protocol_sha256": expected["protocol_sha256"],
        "evidence_manifest": {"key": manifest["key"], "sha256": manifest["sha256"], "bytes": len(raw)},
        "network_volume_attached": False,
        "promotion_performed": False,
        "lineage_mutated": False,
    }


def create_with_retries(execution: RunPodExecutionAdapter, body: dict[str, object]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    last_error: BaseException | None = None
    for attempt in range(1, CREATE_ATTEMPTS + 1):
        try:
            pod = execution._create_pod(body)
            observations.append({"attempt": attempt, "status": "created", "pod_id": pod.get("id"), "gpu": pod.get("gpu")})
            return pod, observations
        except Exception as exc:
            last_error = exc
            observations.append({"attempt": attempt, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            lowered = str(exc).lower()
            if any(marker in lowered for marker in ("402", "insufficient funds", "insufficient credit", "insufficient balance")):
                raise
            if attempt < CREATE_ATTEMPTS:
                time.sleep(CREATE_RETRY_SECONDS)
    raise RuntimeError(f"Diagnostic Scaling Pod creation exhausted retries: {last_error}") from last_error


def render_only(model_id: str, *, repo_sha: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    contract = load_contract()
    candidate = select_candidate(contract, model_id)
    repo_sha = repo_sha or os.environ.get("GITHUB_SHA", "0" * 40)
    run_id = run_id or f"diagnostic-scaling-v1-{os.environ.get('GITHUB_RUN_ID', 'dry-run')}"
    execution_id = f"{run_id}-{slug(model_id)}"
    env = placeholder_environment(repo_sha=repo_sha, run_id=run_id, execution_id=execution_id, attempt=1, model_id=model_id)
    body = build_pod_request(
        contract,
        candidate,
        name=f"hephaestus-{execution_id}-a1",
        env=env,
    )
    return {
        "status": "rendered_not_launched",
        "launch_authorized": False,
        "model_id": model_id,
        "revision": candidate["revision"],
        "protocol_sha256": sha(CONTRACT_PATH.read_bytes()),
        "request": redacted_request(body),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--execute", action="store_true", help="actually create the governed RunPod (requires second authorization lock)")
    args = parser.parse_args()

    if not args.execute:
        rendered = render_only(args.model_id)
        path = ROOT / f"diagnostic_scaling_launch_request_{slug(args.model_id)}.json"
        path.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("DIAGNOSTIC_SCALING_LAUNCH_RENDER_JSON " + json.dumps({
            "status": rendered["status"],
            "model_id": args.model_id,
            "path": str(path),
        }, sort_keys=True), flush=True)
        return 0

    if os.environ.get(AUTH_ENV, "").strip() != "YES":
        raise RuntimeError(f"refusing to create a Pod without {AUTH_ENV}=YES")

    contract = load_contract()
    candidate = select_candidate(contract, args.model_id)
    repo_sha = required("GITHUB_SHA")
    run_id = os.environ.get("HEPHAESTUS_DS_RUN_ID", "").strip() or f"diagnostic-scaling-v1-{required('GITHUB_RUN_ID')}"
    execution_id = f"{run_id}-{slug(args.model_id)}"
    required("RUNPOD_API_KEY")
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = storage.s3_client()
    pod_id: str | None = None
    record: dict[str, Any] = {
        "launcher_version": "diagnostic-scaling-runpod.v1",
        "started_at": now(),
        "run_id": run_id,
        "execution_id": execution_id,
        "model_id": args.model_id,
        "revision": candidate["revision"],
        "repo_sha": repo_sha,
        "status": "starting",
        "network_volume_attached": False,
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
    }
    try:
        verify_admission(client, repo_sha=repo_sha, contract=contract, model_id=args.model_id)
        existing = maybe_existing(client, run_id=run_id, model_id=args.model_id, contract=contract)
        if existing is not None:
            record.update({"status": "verified_existing_terminal", "existing_evidence": existing, "teardown": {"not_created": True, "verified_absent": True}})
            print("DIAGNOSTIC_SCALING_LAUNCH_JSON " + json.dumps({"status": record["status"], "model_id": args.model_id}, sort_keys=True), flush=True)
            return 0

        attempt = next_attempt(client, execution_id)
        record["attempt"] = attempt
        pod_prefix = f"hephaestus-{execution_id}"
        record["prelaunch_cleanup"] = lifecycle.cleanup_experiment_pods(execution, pod_prefix)
        env = pod_environment(repo_sha=repo_sha, run_id=run_id, execution_id=execution_id, attempt=attempt, model_id=args.model_id)
        body = build_pod_request(contract, candidate, name=f"{pod_prefix}-a{attempt}", env=env)
        record["request_redacted"] = redacted_request(body)
        pod, observations = create_with_retries(execution, body)
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        record["capacity_selection"] = observations
        terminal = wait_terminal(client, execution, execution_id=execution_id, attempt=attempt, pod_id=pod_id)
        verification = verify_terminal(client, terminal, run_id=run_id, execution_id=execution_id, model_id=args.model_id, contract=contract)
        record.update({"status": "verified", "verification": verification})
        print("DIAGNOSTIC_SCALING_LAUNCH_JSON " + json.dumps({
            "status": "verified",
            "run_id": run_id,
            "model_id": args.model_id,
            "pod_id": pod_id,
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if pod_id:
            record["teardown"] = lifecycle.delete_pod_with_retries(execution, pod_id)
        else:
            record.setdefault("teardown", {"not_created": True, "verified_absent": True})
        record["completed_at"] = now()
        path = ROOT / f"diagnostic_scaling_launcher_{slug(args.model_id)}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if pod_id and not record["teardown"].get("verified_absent"):
            raise RuntimeError(f"could not verify Diagnostic Scaling Pod teardown: {pod_id}")


if __name__ == "__main__":
    raise SystemExit(main())
