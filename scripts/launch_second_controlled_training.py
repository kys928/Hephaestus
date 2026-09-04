#!/usr/bin/env python3
"""Launch, monitor, independently verify, and tear down the second controlled GPU training run."""
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

RUN_ID = "planned-run-b8e558e54effac85"
EXPERIMENT_ID = "experiment-d0e911d6bd1fb7ae"
PRIMARY_VARIABLE = "dataset_mixture"
APPROVAL_REF = "approval://operator/chat-2026-09-04-second-controlled-experiment"
PROCESSED = "sha256:bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac"
PROCESSED_BYTES = 157_151_627
MODEL_IDENTITY = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
TOKENIZER_IDENTITY = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
MANIFEST_HASH = "sha256:0495018a0cc7c70494d5a00bc51a471568e850d8e3fa11cb0696c9674c71cc76"
CONTRACT_HASH = "sha256:ef273fe913f582289ffad2cd05a431e9d541091a51db97b0a649eb47579f2a5a"
EVIDENCE_HASH = "sha256:d78c9aef9d9522fa0befb77f275ae0b025df1c11dc8b43845098747a96deb0f6"
APPROVAL_HASH = "sha256:9c377cec52d831412f1a716ac756695e21a28a4c59bb6ba55c673733bee7d48e"
RECEIPT_HASH = "sha256:c66bdc6f9d46c425e3ba88ab123de45be52061ea373963ca276c69ebfd2aed37"
INITIAL_WEIGHTS_HASH = "sha256:1bc32006ae216fb4d6b7dd5ce66241e487d552ffcebbbb87c6a029cd22074ce1"
VOLUME_ID = base.VOLUME_ID
DATACENTER_ID = base.DATACENTER_ID
SCIENTIFIC_PREFIX = base.SCIENTIFIC_PREFIX
IMAGE = base.IMAGE
MAX_SECONDS = 1800
POLL_SECONDS = 5


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
    "result_version": "second-controlled-training-result.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ.get("HEPHAESTUS_RUN_ID", "unknown"),
    "status": "pod_bootstrap_failed",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
    "evaluation_performed": False,
    "judge_action_applied": False,
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
"$PY" -m pip install --disable-pip-version-check -e . 'transformers>=4.46,<6' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1'
"$PY" - <<'PYCHECK'
import torch
assert torch.cuda.is_available(), "CUDA unavailable after controlled-training venv bootstrap"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PYCHECK
"$PY" -m py_compile scripts/run_second_controlled_training.py
"$PY" scripts/run_second_controlled_training.py
'''


def create_pod(execution: RunPodExecutionAdapter, repo_sha: str) -> dict[str, Any]:
    return execution.create_bounded_gpu_pod(
        name=f"hephaestus-{RUN_ID}"[:180],
        image_name=IMAGE,
        gpu_type_ids=[
            "NVIDIA GeForce RTX 3070",
            "NVIDIA GeForce RTX 3080",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA L4",
            "NVIDIA GeForce RTX 4090",
        ],
        docker_start_cmd=["bash", "-lc", pod_shell()],
        env={
            "HEPHAESTUS_RUN_ID": RUN_ID,
            "HEPHAESTUS_REPO_SHA": repo_sha,
            "HEPHAESTUS_OPERATOR_APPROVAL_REF": APPROVAL_REF,
            "HEPHAESTUS_MAX_WALL_SECONDS": "1200",
        },
        container_disk_in_gb=20,
        interruptible=False,
    )


def wait_for_result(client: Any, execution: RunPodExecutionAdapter, pod_id: str) -> tuple[dict[str, Any], list[dict[str, object]]]:
    key = f"{SCIENTIFIC_PREFIX}/executions/{RUN_ID}/driver_result.json"
    deadline = time.monotonic() + MAX_SECONDS
    observations: list[dict[str, object]] = []
    last_status: str | None = None
    while time.monotonic() < deadline:
        raw = base.maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("controlled training terminal record is not a JSON object")
            return payload, observations
        pod = base.pod_snapshot(execution, pod_id)
        status = str(pod.get("desiredStatus", "unknown")) if pod else "unknown"
        if status != last_status:
            observations.append({"at": datetime.now(timezone.utc).isoformat(), "desired_status": status})
            last_status = status
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"controlled training result did not appear within {MAX_SECONDS} seconds")


def verify_model_and_tokenizer(client: Any) -> dict[str, object]:
    model_prefix = f"{SCIENTIFIC_PREFIX}/materialized/models/{MODEL_IDENTITY.removeprefix('sha256:')}"
    tokenizer_prefix = f"{SCIENTIFIC_PREFIX}/materialized/tokenizers/{TOKENIZER_IDENTITY.removeprefix('sha256:')}"
    model_checks = [
        base.verify_key(client, f"{model_prefix}/{name}", ref)
        for name, ref in sorted(base.MODEL_COMPONENTS.items())
    ]
    tokenizer_checks = [
        base.verify_key(client, f"{tokenizer_prefix}/{name}", ref)
        for name, ref in sorted(base.TOKENIZER_COMPONENTS.items())
    ]
    if base.directory_identity(base.MODEL_COMPONENTS) != MODEL_IDENTITY:
        raise RuntimeError("random-init model directory identity reconstruction failed")
    if base.directory_identity(base.TOKENIZER_COMPONENTS) != TOKENIZER_IDENTITY:
        raise RuntimeError("tokenizer directory identity reconstruction failed")
    return {
        "model": {"directory_content_identity": MODEL_IDENTITY, "components": model_checks},
        "tokenizer": {"directory_content_identity": TOKENIZER_IDENTITY, "components": tokenizer_checks},
    }


def verify_prepared_inputs(client: Any) -> dict[str, object]:
    prefix = f"{SCIENTIFIC_PREFIX}/runtime_bindings/{RUN_ID}/dataset"
    prepared = {
        "processed": base.verify_key(client, f"{prefix}/trainable.jsonl", PROCESSED, PROCESSED_BYTES),
        "manifest": base.verify_key(client, f"{prefix}/dataset_manifest.json", MANIFEST_HASH),
        "contract": base.verify_key(client, f"{prefix}/trainable_data_contract.json", CONTRACT_HASH),
        "processing_evidence": base.verify_key(client, f"{prefix}/processing_evidence.json", EVIDENCE_HASH),
        "approval": base.verify_key(client, f"{prefix}/dataset_approval.json", APPROVAL_HASH),
        "acquisition_receipt": base.verify_key(client, f"{prefix}/acquisition_receipt.json", RECEIPT_HASH),
    }
    object_key = base.object_key(PROCESSED)
    prepared["content_addressed_processed"] = base.verify_key(client, object_key, PROCESSED, PROCESSED_BYTES)
    return prepared


def list_prefix(client: Any, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": VOLUME_ID, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("Contents", []) or []:
            rows.append({"key": str(item["Key"]), "size": int(item.get("Size", 0))})
        if not page.get("IsTruncated"):
            break
        token = str(page.get("NextContinuationToken") or "")
    rows.sort(key=lambda item: str(item["key"]))
    return rows


def verify_training(client: Any, result: dict[str, Any]) -> dict[str, object]:
    if result.get("status") != "completed":
        raise RuntimeError(f"controlled training did not complete: {result.get('status')}: {result.get('error', '')}")
    if result.get("run_id") != RUN_ID or result.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("controlled training terminal identity drifted")
    if result.get("primary_variable") != PRIMARY_VARIABLE:
        raise RuntimeError("controlled training changed the declared primary variable")
    if result.get("evaluation_performed") is not False or result.get("judge_action_applied") is not False:
        raise RuntimeError("training driver crossed evaluation/Judge application boundary")

    inputs = result.get("verified_inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("controlled training terminal record lacks verified inputs")
    expected_inputs = {
        "processed_dataset_sha256": PROCESSED,
        "tokenizer_directory_identity": TOKENIZER_IDENTITY,
        "model_directory_identity": MODEL_IDENTITY,
        "dataset_manifest_sha256": MANIFEST_HASH,
        "trainable_data_contract_sha256": CONTRACT_HASH,
        "processing_evidence_sha256": EVIDENCE_HASH,
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            raise RuntimeError(f"controlled training input identity drift for {name}")

    recipe = result.get("controlled_recipe")
    if not isinstance(recipe, dict):
        raise RuntimeError("controlled recipe evidence missing")
    expected_recipe = {
        "max_steps": 100,
        "batch_size": 8,
        "context_length": 256,
        "learning_rate": 0.0005,
        "warmup_steps": 10,
        "optimizer": "adamw",
        "scheduler": "linear",
        "weight_decay": 0.01,
        "gradient_clipping": 1.0,
        "checkpoint_every_steps": 100,
        "shuffle": False,
        "dtype": "float32",
        "seed": 1729,
        "only_primary_variable_changed": PRIMARY_VARIABLE,
    }
    for name, expected in expected_recipe.items():
        if recipe.get(name) != expected:
            raise RuntimeError(f"controlled recipe drift for {name}: {recipe.get(name)!r}")

    checkpoint_info = result.get("checkpoint_verification")
    if not isinstance(checkpoint_info, dict) or checkpoint_info.get("valid") is not True:
        raise RuntimeError("training driver did not validate the finalized checkpoint")
    checkpoint = base.verify_checkpoint(client, str(checkpoint_info.get("checkpoint_ref") or ""))
    if checkpoint["checkpoint_manifest_hash"] != checkpoint_info.get("checkpoint_manifest_hash"):
        raise RuntimeError("independent checkpoint manifest verification disagrees")
    if not str(checkpoint.get("checkpoint_prefix", "")).endswith(f"{RUN_ID}/checkpoint_step_100"):
        raise RuntimeError("controlled training did not finalize at step 100")

    trained_weights: str | None = None
    for component in checkpoint.get("components", []):
        if isinstance(component, dict) and str(component.get("key", "")).endswith("/model/model.safetensors"):
            trained_weights = str(component.get("sha256") or "")
            break
    if not trained_weights:
        raise RuntimeError("final checkpoint lacks model.safetensors")
    if trained_weights == INITIAL_WEIGHTS_HASH:
        raise RuntimeError("controlled training did not change model weights")

    run_prefix = f"{SCIENTIFIC_PREFIX}/runs/{RUN_ID}"
    runtime_files: dict[str, object] = {}
    for name in (
        "handle.json",
        "prepared_job.json",
        "normalized_training_config.json",
        "resource_estimate.json",
        "metrics_summary.json",
        "runtime_result.json",
        "final_result.json",
        "checkpoint_record.json",
        "scientific_run_result.json",
    ):
        key = f"{run_prefix}/{name}"
        raw = base.read_key(client, key)
        json.loads(raw.decode("utf-8"))
        runtime_files[name] = {
            "key": key,
            "sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
            "bytes": len(raw),
        }

    metrics_summary = json.loads(base.read_key(client, f"{run_prefix}/metrics_summary.json").decode("utf-8"))
    runtime_result = json.loads(base.read_key(client, f"{run_prefix}/runtime_result.json").decode("utf-8"))
    if runtime_result.get("status") != "completed":
        raise RuntimeError("runtime_result is not completed")
    final_step = int(metrics_summary.get("final_step", metrics_summary.get("step", 0)) or 0)
    if final_step not in {0, 100}:
        raise RuntimeError(f"unexpected final metrics step: {final_step}")

    objects = list_prefix(client, run_prefix + "/")
    if not objects:
        raise RuntimeError("controlled training run inventory is empty")

    return {
        "verification_version": "second-controlled-training-verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "primary_variable": PRIMARY_VARIABLE,
        "input_identities": expected_inputs,
        "checkpoint": checkpoint,
        "initial_model_weights_sha256": INITIAL_WEIGHTS_HASH,
        "trained_model_weights_sha256": trained_weights,
        "weights_changed": trained_weights != INITIAL_WEIGHTS_HASH,
        "runtime_files": runtime_files,
        "metrics_summary": metrics_summary,
        "runtime_result": runtime_result,
        "run_object_count": len(objects),
        "run_total_bytes": sum(int(item["size"]) for item in objects),
        "run_inventory_hash": base.canonical_hash(objects),
        "run_objects": objects,
        "terminal_result": result,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    base.required("RUNPOD_API_KEY")
    repo_sha = base.required("GITHUB_SHA")
    client = base.s3_client()
    client.head_bucket(Bucket=VOLUME_ID)
    model_tokenizer = verify_model_and_tokenizer(client)
    prepared = verify_prepared_inputs(client)
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())

    started = time.monotonic()
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
        pod = create_pod(execution, repo_sha)
        pod_id = str(pod["id"])
        teardown = {"deleted": False, "not_created": False}
        result, observations = wait_for_result(client, execution, pod_id)
        final_pod = base.pod_snapshot(execution, pod_id)
        verification = verify_training(client, result)
    except BaseException as exc:
        launcher_error = f"{type(exc).__name__}: {exc}"
        lowered = launcher_error.lower()
        funds_unavailable = "http 402" in lowered or any(
            phrase in lowered
            for phrase in ("insufficient funds", "insufficient credit", "insufficient balance", "not enough credit")
        )
        if pod_id:
            final_pod = base.pod_snapshot(execution, pod_id)
    finally:
        if pod_id:
            teardown = base.delete_pod(execution, pod_id)

    elapsed = time.monotonic() - started
    pod_view = final_pod or pod or {}
    cost_per_hr = pod_view.get("adjustedCostPerHr", pod_view.get("costPerHr")) if isinstance(pod_view, dict) else None
    try:
        estimated_cost = float(cost_per_hr) * elapsed / 3600.0 if cost_per_hr is not None else None
    except (TypeError, ValueError):
        estimated_cost = None

    launcher = {
        "launcher_version": "second-controlled-training-launcher.v1",
        "created_at": created_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "repo_sha": repo_sha,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "image": IMAGE,
        "operator_approval_ref": APPROVAL_REF,
        "primary_variable": PRIMARY_VARIABLE,
        "pod_id": pod_id,
        "pod": pod_view,
        "pod_status_observations": observations,
        "elapsed_seconds": elapsed,
        "cost_per_hour": cost_per_hr,
        "estimated_cost_for_observed_elapsed_time": estimated_cost,
        "estimated_cost_is_not_billing_record": True,
        "funds_unavailable": funds_unavailable,
        "teardown": teardown,
        "model_tokenizer_verified_before_launch": model_tokenizer,
        "prepared_data_verified_before_launch": prepared,
        "terminal_result": result,
        "verification_status": verification.get("status") if verification else None,
        "error": launcher_error,
    }
    atomic_json(Path("second_controlled_training_launcher.json"), launcher)
    if verification is not None:
        verification["launcher"] = {
            "pod_id": pod_id,
            "repo_sha": repo_sha,
            "elapsed_seconds": elapsed,
            "cost_per_hour": cost_per_hr,
            "estimated_cost_for_observed_elapsed_time": estimated_cost,
            "teardown": teardown,
        }
        atomic_json(Path("second_controlled_training_verification.json"), verification)

    if launcher_error:
        print(json.dumps(launcher, indent=2, sort_keys=True))
        return 1
    if not verification or verification.get("status") != "verified":
        print(json.dumps(launcher, indent=2, sort_keys=True))
        return 1
    print(json.dumps({
        "status": "verified",
        "run_id": RUN_ID,
        "checkpoint_manifest_hash": verification["checkpoint"]["checkpoint_manifest_hash"],
        "trained_model_weights_sha256": verification["trained_model_weights_sha256"],
        "weights_changed": verification["weights_changed"],
        "pod_deleted": teardown.get("deleted"),
        "funds_unavailable": funds_unavailable,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
