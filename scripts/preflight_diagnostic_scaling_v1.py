#!/usr/bin/env python3
"""Build and persist the launch admission package for Diagnostic Scaling V1.

This gate performs no GPU or RunPod Pod work. It validates the frozen experiment,
reuses the immutable Test-3 training records, writes a cohort-specific dependency
lock, and persists all launch inputs to RunPod S3 with byte-for-byte readback.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_first_bounded_scientific_training as storage
import validate_diagnostic_scaling_v1_contract as validator

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json"
BASE_DISTANCE_PATH = ROOT / "configs/experiments/hephaestus_distance_to_role_v1.json"
OUT = ROOT / "diagnostic_scaling_admission"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_s3(client: Any, key: str) -> bytes:
    return storage.read_key(client, key)


def requirements_bytes(contract: dict[str, Any]) -> bytes:
    deps = contract["execution"]["runtime_dependencies"]
    lines = [
        f"transformers=={deps['transformers']}",
        f"accelerate=={deps['accelerate']}",
        f"safetensors=={deps['safetensors']}",
        f"huggingface-hub=={deps['huggingface_hub']}",
        f"peft=={deps['peft']}",
        "boto3>=1.35,<2",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_admission(
    *,
    repo_sha: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    base_distance_raw: bytes,
    training_raw: bytes,
    contamination_raw: bytes,
    requirements_raw: bytes,
    remote_rows: list[dict[str, Any]],
    runtime_compatibility: dict[str, Any],
    source_training_key: str,
) -> dict[str, Any]:
    contamination = json.loads(contamination_raw)
    if contamination.get("passed") is not True:
        raise RuntimeError("source Test-3 training data no longer has passing contamination evidence")
    dataset_sha = sha(training_raw)
    if dataset_sha != contract["sources"]["source_training_dataset_sha256"]:
        raise RuntimeError(f"source training dataset hash drifted: {dataset_sha}")
    rows = [json.loads(line) for line in training_raw.decode("utf-8").splitlines() if line.strip()]
    roles = [*contract["roles"]["primary"], *contract["roles"]["calibration"]]
    expected = int(contract["training"]["examples_per_role"])
    role_counts = {role: sum(1 for row in rows if row.get("role") == role) for role in roles}
    if role_counts != {role: expected for role in roles}:
        raise RuntimeError(f"diagnostic-scaling source role counts drifted: {role_counts}")

    candidate_order = [row["model_id"] for row in contract["candidates"]]
    remote_order = [row["model_id"] for row in remote_rows]
    if remote_order != candidate_order or not all(row.get("remote_revision_verified") for row in remote_rows):
        raise RuntimeError("remote model verification does not exactly cover the frozen cohort")

    return {
        "admission_version": "diagnostic-scaling-v1-admission.v1",
        "status": "launch_inputs_admitted",
        "repo_sha": repo_sha,
        "protocol_id": contract["protocol_id"],
        "protocol_sha256": sha(contract_raw),
        "base_distance_protocol_sha256": sha(base_distance_raw),
        "training_dataset_sha256": dataset_sha,
        "source_training_key": source_training_key,
        "source_contamination_sha256": sha(contamination_raw),
        "runtime_requirements_sha256": sha(requirements_raw),
        "candidate_order": candidate_order,
        "candidate_revisions": {row["model_id"]: row["revision"] for row in contract["candidates"]},
        "remote_models": remote_rows,
        "runtime_compatibility": runtime_compatibility,
        "roles": roles,
        "role_counts": role_counts,
        "dose_optimizer_steps": contract["training"]["dose_optimizer_steps"],
        "container_disk_gb": contract["execution"]["container_disk_gb"],
        "network_volume_attached": False,
        "persistent_model_cache_allowed": False,
        "training_approved": True,
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
        "model_revision_substitution_allowed": False,
    }


def main() -> int:
    repo_sha = required("GITHUB_SHA")
    contract_raw = CONTRACT_PATH.read_bytes()
    base_distance_raw = BASE_DISTANCE_PATH.read_bytes()
    contract = json.loads(contract_raw)
    local = validator.validate_contract(contract, ROOT)
    remote_rows, runtime_compatibility = validator.validate_remote_models(contract)
    if local["status"] != "local_contract_valid":
        raise RuntimeError("contract local validation did not pass")

    base_distance = json.loads(base_distance_raw)
    source_sha = base_distance["sources"]["elasticity_admission_repo_sha"]
    source_prefix = f"{storage.SCIENTIFIC_PREFIX}/adaptation_elasticity/model_admission/{source_sha}"
    source_training_key = f"{source_prefix}/training.jsonl"
    source_contamination_key = f"{source_prefix}/contamination_report.json"

    s3 = storage.s3_client()
    s3.head_bucket(Bucket=storage.VOLUME_ID)
    training_raw = read_s3(s3, source_training_key)
    contamination_raw = read_s3(s3, source_contamination_key)
    requirements_raw = requirements_bytes(contract)

    admission = build_admission(
        repo_sha=repo_sha,
        contract=contract,
        contract_raw=contract_raw,
        base_distance_raw=base_distance_raw,
        training_raw=training_raw,
        contamination_raw=contamination_raw,
        requirements_raw=requirements_raw,
        remote_rows=remote_rows,
        runtime_compatibility=runtime_compatibility,
        source_training_key=source_training_key,
    )
    remote_raw = (json.dumps({"models": remote_rows, "runtime": runtime_compatibility}, indent=2, sort_keys=True) + "\n").encode()
    admission_raw = (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode()

    OUT.mkdir(exist_ok=True)
    objects = {
        "admission.json": admission_raw,
        "protocol.json": contract_raw,
        "training.jsonl": training_raw,
        "contamination_report.json": contamination_raw,
        "requirements.txt": requirements_raw,
        "remote_models.json": remote_raw,
    }
    for name, raw in objects.items():
        (OUT / name).write_bytes(raw)

    destination_prefix = f"{storage.SCIENTIFIC_PREFIX}/diagnostic_scaling/model_admission/{repo_sha}"
    persisted: list[dict[str, Any]] = []
    for name, raw in objects.items():
        key = f"{destination_prefix}/{name}"
        s3.put_object(Bucket=storage.VOLUME_ID, Key=key, Body=raw)
        observed = read_s3(s3, key)
        if observed != raw:
            raise RuntimeError(f"Diagnostic Scaling admission S3 readback mismatch: {key}")
        row = {"key": key, "sha256": sha(raw), "bytes": len(raw)}
        persisted.append(row)
        print("DIAGNOSTIC_SCALING_ADMISSION_OBJECT_JSON " + json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "status": "launch_inputs_admitted",
        "repo_sha": repo_sha,
        "protocol_sha256": admission["protocol_sha256"],
        "training_dataset_sha256": admission["training_dataset_sha256"],
        "candidate_count": len(admission["candidate_order"]),
        "roles": admission["roles"],
        "network_volume_attached": False,
        "s3_prefix": destination_prefix,
        "persisted_objects": persisted,
    }
    (OUT / "admission_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DIAGNOSTIC_SCALING_ADMISSION_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
