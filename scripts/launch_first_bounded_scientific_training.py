#!/usr/bin/env python3
"""Launch one bounded RunPod GPU training Pod and independently verify its volume outputs.

This script runs from GitHub Actions. It is the external execution boundary:
- creates one on-demand Secure Cloud GPU Pod in the Network Volume's datacenter;
- attaches cviwpryzao at /workspace;
- runs the repository's governed first-scientific-training driver at the exact commit;
- waits for the volume-resident terminal record;
- re-reads and hashes the resulting checkpoint through RunPod S3; and
- deletes the Pod in a finally block regardless of outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

VOLUME_ID = "cviwpryzao"
DATACENTER_ID = "EU-CZ-1"
SCIENTIFIC_PREFIX = "hephaestus/scientific/v1"
PROCESSED_DATA = "sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c"
TOKENIZER_IDENTITY = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
MODEL_IDENTITY = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
BOOTSTRAP_BUNDLE = "sha256:6774e92d2b595353a18211ffa772fb82b362d462ee2e3c26144f705d26525436"
IMAGE = "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
MAX_LAUNCH_SECONDS = 1800
POLL_SECONDS = 5

MODEL_COMPONENTS = {
    "config.json": "sha256:e4bae2d91ba1c40722209bf43fadb83dff7260cfb7d39f7d37d89a0e500ced80",
    "generation_config.json": "sha256:638dbab69ef55c5387c925da4027750ca5a38f6fa1aa9e99f263b06881261f26",
    "model.safetensors": "sha256:1bc32006ae216fb4d6b7dd5ce66241e487d552ffcebbbb87c6a029cd22074ce1",
}
TOKENIZER_COMPONENTS = {
    "tokenizer.json": "sha256:3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25",
    "tokenizer_config.json": "sha256:ddffe31f2ecc9a35891ac1d9c250cf9dc019cb9c23650a5fd6eeea3e89aab891",
}
APPROVAL_REF = "approval://operator/explicit-request-2026-09-04-first-bounded-scientific-training"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def digest(ref: str) -> str:
    return ref.removeprefix("sha256:")


def canonical_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def directory_identity(components: dict[str, str]) -> str:
    raw = json.dumps(
        {name: digest(value) for name, value in components.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )


def read_key(client: Any, key: str) -> bytes:
    response = client.get_object(Bucket=VOLUME_ID, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def maybe_read_key(client: Any, key: str) -> bytes | None:
    try:
        return read_key(client, key)
    except Exception as exc:  # noqa: BLE001
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def verify_key(client: Any, key: str, expected: str, expected_size: int | None = None) -> dict[str, object]:
    raw = read_key(client, key)
    observed = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if observed != expected:
        raise RuntimeError(f"S3 SHA mismatch for {key}: {observed} != {expected}")
    if expected_size is not None and len(raw) != expected_size:
        raise RuntimeError(f"S3 size mismatch for {key}: {len(raw)} != {expected_size}")
    return {"key": key, "sha256": observed, "bytes": len(raw)}


def object_key(ref: str) -> str:
    d = digest(ref)
    return f"{SCIENTIFIC_PREFIX}/objects/sha256/{d[:2]}/{d}"


def verify_bootstrap_inputs(client: Any) -> dict[str, object]:
    verified = {
        "bootstrap_bundle": verify_key(client, object_key(BOOTSTRAP_BUNDLE), BOOTSTRAP_BUNDLE),
        "processed_dataset": verify_key(client, object_key(PROCESSED_DATA), PROCESSED_DATA, 15_920_505),
    }
    model_prefix = f"{SCIENTIFIC_PREFIX}/materialized/models/{digest(MODEL_IDENTITY)}"
    tokenizer_prefix = f"{SCIENTIFIC_PREFIX}/materialized/tokenizers/{digest(TOKENIZER_IDENTITY)}"
    model_checks = [verify_key(client, f"{model_prefix}/{name}", ref) for name, ref in sorted(MODEL_COMPONENTS.items())]
    tokenizer_checks = [verify_key(client, f"{tokenizer_prefix}/{name}", ref) for name, ref in sorted(TOKENIZER_COMPONENTS.items())]
    rebuilt_model = directory_identity(MODEL_COMPONENTS)
    rebuilt_tokenizer = directory_identity(TOKENIZER_COMPONENTS)
    if rebuilt_model != MODEL_IDENTITY or rebuilt_tokenizer != TOKENIZER_IDENTITY:
        raise RuntimeError("independent bootstrap directory identity reconstruction failed")
    verified["model"] = {"directory_content_identity": rebuilt_model, "components": model_checks}
    verified["tokenizer"] = {"directory_content_identity": rebuilt_tokenizer, "components": tokenizer_checks}
    return verified


def pod_shell() -> str:
    return r'''set -Eeuo pipefail
RUN_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_RUN_ID}"
mkdir -p "$RUN_DIR"
exec >"$RUN_DIR/pod_runtime.log" 2>&1
write_bootstrap_failure() {
  code=$?
  if [ "$code" -ne 0 ] && [ ! -f "$RUN_DIR/driver_result.json" ]; then
    python - "$RUN_DIR/driver_result.json" "$code" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
payload = {
    "result_version": "first-bounded-scientific-training-result.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ.get("HEPHAESTUS_RUN_ID", "unknown"),
    "status": "pod_bootstrap_failed",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
}
with open(path + ".partial", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
os.replace(path + ".partial", path)
PYFAIL
  fi
}
trap write_bootstrap_failure EXIT
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
rm -rf /opt/hephaestus-src
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m pip install --disable-pip-version-check -e . 'transformers>=4.46,<6' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1'
python -m py_compile scripts/run_first_bounded_scientific_training.py
python scripts/run_first_bounded_scientific_training.py
'''


def create_pod(execution: RunPodExecutionAdapter, run_id: str, repo_sha: str) -> dict[str, Any]:
    gpu_type_ids = [
        "Tesla T4",
        "NVIDIA RTX A2000",
        "NVIDIA RTX 2000 Ada Generation",
        "NVIDIA RTX A4000",
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA L4",
        "NVIDIA GeForce RTX 4090",
    ]
    return execution.create_bounded_gpu_pod(
        name=f"hephaestus-{run_id}"[:180],
        image_name=IMAGE,
        gpu_type_ids=gpu_type_ids,
        docker_start_cmd=["bash", "-lc", pod_shell()],
        env={
            "HEPHAESTUS_RUN_ID": run_id,
            "HEPHAESTUS_REPO_SHA": repo_sha,
            "HEPHAESTUS_OPERATOR_APPROVAL_REF": APPROVAL_REF,
            "HEPHAESTUS_MAX_WALL_SECONDS": "1200",
        },
        container_disk_in_gb=20,
        interruptible=False,
    )


def pod_snapshot(execution: RunPodExecutionAdapter, pod_id: str) -> dict[str, Any] | None:
    try:
        return execution.get_pod(pod_id)
    except Exception:
        return None


def delete_pod(execution: RunPodExecutionAdapter, pod_id: str) -> dict[str, object]:
    try:
        execution.delete_pod(pod_id)
        return {"deleted": True, "status": 204}
    except Exception as exc:  # noqa: BLE001
        return {"deleted": False, "error": f"{type(exc).__name__}: {exc}"}


def wait_for_result(client: Any, execution: RunPodExecutionAdapter, run_id: str, pod_id: str) -> tuple[dict[str, Any], list[dict[str, object]]]:
    key = f"{SCIENTIFIC_PREFIX}/executions/{run_id}/driver_result.json"
    deadline = time.monotonic() + MAX_LAUNCH_SECONDS
    observations: list[dict[str, object]] = []
    last_status: str | None = None
    while time.monotonic() < deadline:
        raw = maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("driver terminal record is not a JSON object")
            return payload, observations
        pod = pod_snapshot(execution, pod_id)
        status = str(pod.get("desiredStatus", "unknown")) if pod else "unknown"
        if status != last_status:
            observations.append({"at": datetime.now(timezone.utc).isoformat(), "desired_status": status})
            last_status = status
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"training result did not appear within {MAX_LAUNCH_SECONDS} seconds")


def verify_checkpoint(client: Any, checkpoint_path: str) -> dict[str, object]:
    prefix = checkpoint_path.removeprefix("/workspace/").rstrip("/")
    if prefix == checkpoint_path:
        raise RuntimeError(f"checkpoint is not on expected /workspace volume: {checkpoint_path}")
    manifest_key = f"{prefix}/checkpoint_manifest.json"
    manifest_raw = read_key(client, manifest_key)
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("partial_write") is not False:
        raise RuntimeError("checkpoint manifest is missing or not finalized")
    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        raise RuntimeError("checkpoint manifest has no components")
    expected_manifest_hash = canonical_hash(components)
    if manifest.get("manifest_hash") != expected_manifest_hash:
        raise RuntimeError("checkpoint canonical manifest hash mismatch")
    checked: list[dict[str, object]] = []
    for relative, expected in sorted(components.items()):
        checked.append(verify_key(client, f"{prefix}/{relative}", str(expected)))
    return {
        "checkpoint_prefix": prefix,
        "manifest_key": manifest_key,
        "checkpoint_manifest_hash": expected_manifest_hash,
        "component_count": len(checked),
        "components": checked,
    }


def verify_run_outputs(client: Any, run_id: str, result: dict[str, Any]) -> dict[str, object]:
    if result.get("status") != "completed":
        raise RuntimeError(f"training did not complete: {result.get('status')}: {result.get('error', '')}")
    inputs = result.get("verified_inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("terminal result lacks verified input identities")
    expected_inputs = {
        "processed_dataset_sha256": PROCESSED_DATA,
        "tokenizer_directory_identity": TOKENIZER_IDENTITY,
        "model_directory_identity": MODEL_IDENTITY,
        "bootstrap_bundle_artifact_ref": BOOTSTRAP_BUNDLE,
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            raise RuntimeError(f"terminal input identity drift for {name}")

    checkpoint_info = result.get("checkpoint_verification")
    if not isinstance(checkpoint_info, dict) or checkpoint_info.get("valid") is not True:
        raise RuntimeError("driver did not validate its finalized checkpoint")
    independent_checkpoint = verify_checkpoint(client, str(checkpoint_info.get("checkpoint_ref") or ""))
    if independent_checkpoint["checkpoint_manifest_hash"] != checkpoint_info.get("checkpoint_manifest_hash"):
        raise RuntimeError("independent checkpoint manifest hash disagrees with lifecycle")

    run_prefix = f"{SCIENTIFIC_PREFIX}/runs/{run_id}"
    runtime_files: dict[str, object] = {}
    for name in (
        "handle.json", "prepared_job.json", "normalized_training_config.json",
        "resource_estimate.json", "metrics_summary.json", "runtime_result.json",
        "final_result.json", "checkpoint_record.json", "scientific_run_result.json",
    ):
        key = f"{run_prefix}/{name}"
        raw = read_key(client, key)
        runtime_files[name] = {
            "key": key,
            "sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
            "bytes": len(raw),
        }
        json.loads(raw.decode("utf-8"))

    binding_dataset_key = f"{SCIENTIFIC_PREFIX}/runtime_bindings/{run_id}/dataset/trainable.jsonl"
    binding_dataset = verify_key(client, binding_dataset_key, PROCESSED_DATA, 15_920_505)

    listed: list[dict[str, object]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": VOLUME_ID, "Prefix": run_prefix + "/", "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("Contents", []) or []:
            listed.append({"key": str(item["Key"]), "size": int(item.get("Size", 0))})
        if not page.get("IsTruncated"):
            break
        token = str(page.get("NextContinuationToken") or "")
    listed.sort(key=lambda item: str(item["key"]))

    return {
        "verification_version": "first-bounded-scientific-training-verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "run_id": run_id,
        "input_identities": expected_inputs,
        "runtime_bound_processed_dataset": binding_dataset,
        "checkpoint": independent_checkpoint,
        "runtime_files": runtime_files,
        "run_object_count": len(listed),
        "run_total_bytes": sum(int(item["size"]) for item in listed),
        "run_inventory_hash": canonical_hash(listed),
        "run_objects": listed,
        "terminal_result": result,
    }


def main() -> int:
    required("RUNPOD_API_KEY")
    repo_sha = required("GITHUB_SHA")
    workflow_run = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
    run_id = f"first-bounded-scientific-training-001-{workflow_run}"
    client = s3_client()
    client.head_bucket(Bucket=VOLUME_ID)
    bootstrap_verification = verify_bootstrap_inputs(client)
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())

    launch_started = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    pod: dict[str, Any] | None = None
    pod_id: str | None = None
    final_pod: dict[str, Any] | None = None
    teardown: dict[str, object] = {"deleted": False, "not_created": True}
    result: dict[str, Any] | None = None
    verification: dict[str, object] | None = None
    observations: list[dict[str, object]] = []
    launcher_error: str | None = None
    funds_unavailable = False

    try:
        pod = create_pod(execution, run_id, repo_sha)
        pod_id = str(pod["id"])
        teardown = {"deleted": False, "not_created": False}
        result, observations = wait_for_result(client, execution, run_id, pod_id)
        final_pod = pod_snapshot(execution, pod_id)
        verification = verify_run_outputs(client, run_id, result)
    except BaseException as exc:  # noqa: BLE001
        launcher_error = f"{type(exc).__name__}: {exc}"
        lowered = launcher_error.lower()
        funds_unavailable = "http 402" in lowered or any(
            word in lowered for word in ("insufficient funds", "insufficient credit", "insufficient balance", "not enough credit")
        )
        if pod_id:
            final_pod = pod_snapshot(execution, pod_id)
    finally:
        if pod_id:
            teardown = delete_pod(execution, pod_id)

    elapsed = time.monotonic() - launch_started
    pod_view = final_pod or pod or {}
    cost_per_hr = pod_view.get("adjustedCostPerHr", pod_view.get("costPerHr")) if isinstance(pod_view, dict) else None
    try:
        estimated_cost = float(cost_per_hr) * elapsed / 3600.0 if cost_per_hr is not None else None
    except (TypeError, ValueError):
        estimated_cost = None

    launcher = {
        "launcher_version": "first-bounded-scientific-training-launcher.v1",
        "created_at": created_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "repo_sha": repo_sha,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "image": IMAGE,
        "pod_id": pod_id,
        "pod": pod_view,
        "pod_status_observations": observations,
        "elapsed_seconds": elapsed,
        "cost_per_hour": cost_per_hr,
        "estimated_cost_for_observed_elapsed_time": estimated_cost,
        "estimated_cost_is_not_billing_record": True,
        "funds_unavailable": funds_unavailable,
        "teardown": teardown,
        "bootstrap_inputs_verified_before_launch": bootstrap_verification,
        "terminal_result": result,
        "verification_status": verification.get("status") if verification else None,
        "error": launcher_error,
    }
    Path("first_bounded_scientific_training_launcher.json").write_text(
        json.dumps(launcher, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if verification is not None:
        verification["launcher"] = {
            "pod_id": pod_id,
            "gpu": pod_view.get("gpu") if isinstance(pod_view, dict) else None,
            "machine": pod_view.get("machine") if isinstance(pod_view, dict) else None,
            "cost_per_hour": cost_per_hr,
            "elapsed_seconds": elapsed,
            "estimated_cost_for_observed_elapsed_time": estimated_cost,
            "teardown": teardown,
            "repo_sha": repo_sha,
        }
        Path("first_bounded_scientific_training_verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps({
        "run_id": run_id,
        "pod_id": pod_id,
        "training_status": result.get("status") if result else None,
        "verification_status": verification.get("status") if verification else None,
        "checkpoint_manifest_hash": (
            verification.get("checkpoint", {}).get("checkpoint_manifest_hash")
            if isinstance(verification, dict) and isinstance(verification.get("checkpoint"), dict)
            else None
        ),
        "cost_per_hour": cost_per_hr,
        "elapsed_seconds": elapsed,
        "estimated_cost": estimated_cost,
        "funds_unavailable": funds_unavailable,
        "pod_deleted": teardown.get("deleted"),
        "error": launcher_error,
    }, sort_keys=True))
    return 0 if verification is not None and teardown.get("deleted") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
