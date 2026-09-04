#!/usr/bin/env python3
"""Independently verify the first selected-dataset preparation on RunPod S3."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

PROOF = Path("first_selected_dataset_preparation.json")
OUTPUT = Path("first_selected_dataset_preparation_verification.json")
EXPECTED_SELECTION_ID = "dataset-selection-fd8699f8cbd8b4957ca2"
EXPECTED_CANDIDATE_ID = "dataset-fb91684d87fe5f28"
EXPECTED_DATASET_ID = "sail/symbolic-instruction-tuning"
EXPECTED_REVISION = "c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c"
EXPECTED_EXPERIMENT_ID = "experiment-d0e911d6bd1fb7ae"
EXPECTED_RUN_ID = "planned-run-b8e558e54effac85"
EXPECTED_APPROVAL_REF = "approval://operator/chat-2026-09-04-selected-dataset-preparation"
TOKENIZER_REF = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
STORE_PREFIX = "hephaestus/scientific/v1"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 8, "mode": "standard"}),
    )


def stream_hash(s3: Any, bucket: str, key: str) -> tuple[str, int]:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            block = body.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    finally:
        body.close()
    return digest.hexdigest(), size


def read_json(s3: Any, bucket: str, key: str) -> dict[str, object]:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        payload = json.loads(body.read().decode("utf-8"))
    finally:
        body.close()
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {key}")
    return {str(k): v for k, v in payload.items()}


def verify_named(s3: Any, bucket: str, label: str, record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        raise RuntimeError(f"runtime materialization {label} is missing")
    key = str(record.get("key") or "")
    expected = str(record.get("sha256") or "").removeprefix("sha256:")
    expected_size = int(record.get("byte_size") or -1)
    if not key or len(expected) != 64 or expected_size < 0:
        raise RuntimeError(f"runtime materialization {label} has invalid identity evidence")
    observed, size = stream_hash(s3, bucket, key)
    if observed != expected or size != expected_size:
        raise RuntimeError(f"runtime materialization {label} failed independent SHA-256/size verification")
    head = s3.head_object(Bucket=bucket, Key=key)
    metadata = head.get("Metadata")
    if not isinstance(metadata, dict) or metadata.get("sha256") != expected:
        raise RuntimeError(f"runtime materialization {label} has incorrect SHA-256 metadata")
    return {"key": key, "sha256": f"sha256:{observed}", "byte_size": size}


def main() -> None:
    payload = json.loads(PROOF.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("preparation proof is not a JSON object")
    expected_identity = {
        "status": "completed",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "run_id": EXPECTED_RUN_ID,
        "primary_variable": "dataset_mixture",
        "selection_decision_id": EXPECTED_SELECTION_ID,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "dataset_id": EXPECTED_DATASET_ID,
        "revision": EXPECTED_REVISION,
        "license": "mit",
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"proof identity mismatch for {key}: {payload.get(key)!r} != {expected!r}")
    if payload.get("training_launched") is not False or payload.get("model_mutated") is not False or payload.get("experiment_executed") is not False:
        raise RuntimeError("preparation crossed the training/model execution boundary")

    approval = payload.get("approval")
    if not isinstance(approval, dict):
        raise RuntimeError("approval evidence missing")
    if approval.get("selection_decision_id") != EXPECTED_SELECTION_ID:
        raise RuntimeError("approval selection identity mismatch")
    if approval.get("approved_candidate_ids") != [EXPECTED_CANDIDATE_ID]:
        raise RuntimeError("approval candidate identity mismatch")
    if EXPECTED_APPROVAL_REF not in list(approval.get("approval_refs") or []):
        raise RuntimeError("explicit operator approval reference missing")
    if "dataset_selection_approval" not in list(approval.get("approved_requirements") or []):
        raise RuntimeError("ExperimentProposal dataset selection approval was not covered")

    receipt = payload.get("acquisition_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("acquisition receipt missing")
    if receipt.get("completion_status") != "completed":
        raise RuntimeError("acquisition receipt is not completed")
    if receipt.get("dataset_id") != EXPECTED_DATASET_ID or receipt.get("resolved_revision") != EXPECTED_REVISION:
        raise RuntimeError("acquisition receipt immutable dataset identity mismatch")
    if receipt.get("license") != "mit" or receipt.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise RuntimeError("acquisition receipt license/candidate mismatch")
    files = receipt.get("acquired_files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise RuntimeError("expected exactly one bounded acquired shard")
    raw = files[0]
    raw_ref = str(raw.get("artifact_ref") or "")
    raw_hash = str(raw.get("local_content_hash") or "").removeprefix("sha256:")
    raw_size = int(raw.get("size_bytes") or -1)
    if raw_ref != f"sha256:{raw_hash}" or len(raw_hash) != 64 or raw_size <= 0:
        raise RuntimeError("raw acquisition content identity is incomplete")

    s3 = client()
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    s3.head_bucket(Bucket=bucket)
    raw_key = f"{STORE_PREFIX}/objects/sha256/{raw_hash[:2]}/{raw_hash}"
    observed_raw_hash, observed_raw_size = stream_hash(s3, bucket, raw_key)
    if observed_raw_hash != raw_hash or observed_raw_size != raw_size:
        raise RuntimeError("raw acquired shard failed independent durable readback")

    runtime = payload.get("runtime_materialization")
    if not isinstance(runtime, dict):
        raise RuntimeError("runtime materialization evidence missing")
    required_materializations = (
        "processed",
        "trainable_data_contract",
        "dataset_manifest",
        "preprocessing_report",
        "processing_evidence",
        "acquisition_receipt",
        "approval",
    )
    verified_materializations = {
        label: verify_named(s3, bucket, label, runtime.get(label))
        for label in required_materializations
    }

    contract_key = verified_materializations["trainable_data_contract"]["key"]
    manifest_key = verified_materializations["dataset_manifest"]["key"]
    evidence_key = verified_materializations["processing_evidence"]["key"]
    persisted_contract = read_json(s3, bucket, str(contract_key))
    persisted_manifest = read_json(s3, bucket, str(manifest_key))
    persisted_evidence = read_json(s3, bucket, str(evidence_key))
    proof_contract = payload.get("trainable_data_contract")
    if not isinstance(proof_contract, dict) or persisted_contract != proof_contract:
        raise RuntimeError("persisted TrainableDataContract disagrees with proof")
    if persisted_contract.get("run_id") != EXPECTED_RUN_ID:
        raise RuntimeError("TrainableDataContract run identity mismatch")
    if persisted_contract.get("manifest_id") != persisted_manifest.get("manifest_id"):
        raise RuntimeError("TrainableDataContract manifest identity mismatch")
    processed_storage = str(persisted_contract.get("processed_dataset_ref") or "")
    durable_processed = payload.get("durable_records", {}).get("processed") if isinstance(payload.get("durable_records"), dict) else None
    if not isinstance(durable_processed, dict) or processed_storage != str(durable_processed.get("storage_path") or ""):
        raise RuntimeError("TrainableDataContract does not bind the durable content-addressed processed dataset")
    if persisted_evidence.get("tokenizer_compatibility", {}).get("tokenizer_ref") != TOKENIZER_REF:
        raise RuntimeError("processing evidence tokenizer identity mismatch")
    tokenizer_status = persisted_evidence.get("tokenizer_compatibility", {})
    if not isinstance(tokenizer_status, dict) or tokenizer_status.get("status") != "checked" or tokenizer_status.get("compatible") is not True:
        raise RuntimeError("processed dataset tokenizer compatibility was not positively measured")

    processed_key = str(verified_materializations["processed"]["key"])
    response = s3.get_object(Bucket=bucket, Key=processed_key)
    body = response["Body"]
    record_count = 0
    prompt_target_count = 0
    try:
        for raw_line in body.iter_lines():
            if not raw_line:
                continue
            record = json.loads(raw_line.decode("utf-8"))
            if not isinstance(record, dict) or not str(record.get("text") or "").strip():
                raise RuntimeError("processed dataset contains a non-trainable record")
            text = str(record["text"])
            record_count += 1
            if "<|prompt|>" in text and "<|target|>" in text:
                prompt_target_count += 1
    finally:
        body.close()
    if record_count <= 0 or prompt_target_count <= 0:
        raise RuntimeError("processed dataset did not preserve instruction prompt/target wrappers")

    processing = payload.get("preprocessing")
    if not isinstance(processing, dict):
        raise RuntimeError("preprocessing proof missing")
    processing_evidence = processing.get("processing_evidence")
    if not isinstance(processing_evidence, dict):
        raise RuntimeError("processing evidence missing")
    sample_validation = processing_evidence.get("sample_validation")
    if not isinstance(sample_validation, dict):
        raise RuntimeError("sample validation evidence missing")

    verification = {
        "status": "verified",
        "selection_decision_id": EXPECTED_SELECTION_ID,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "dataset_id": EXPECTED_DATASET_ID,
        "resolved_revision": EXPECTED_REVISION,
        "approval_ref": EXPECTED_APPROVAL_REF,
        "acquisition_receipt_id": receipt.get("receipt_id"),
        "raw_shard": {
            "relative_path": raw.get("relative_path"),
            "sha256": f"sha256:{observed_raw_hash}",
            "byte_size": observed_raw_size,
            "durable_key": raw_key,
        },
        "processed": {
            "sha256": verified_materializations["processed"]["sha256"],
            "byte_size": verified_materializations["processed"]["byte_size"],
            "record_count": record_count,
            "prompt_target_record_count": prompt_target_count,
        },
        "trainable_data_contract": {
            "contract_id": persisted_contract.get("contract_id"),
            "manifest_id": persisted_contract.get("manifest_id"),
            "run_id": persisted_contract.get("run_id"),
            "processed_dataset_ref": persisted_contract.get("processed_dataset_ref"),
            "sha256": verified_materializations["trainable_data_contract"]["sha256"],
        },
        "manifest": {
            "manifest_id": persisted_manifest.get("manifest_id"),
            "sha256": verified_materializations["dataset_manifest"]["sha256"],
        },
        "sample_validation": sample_validation,
        "tokenizer_compatibility": tokenizer_status,
        "runtime_materializations": verified_materializations,
        "training_launched": False,
        "model_mutated": False,
        "experiment_executed": False,
    }
    OUTPUT.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("status=verified")
    print(f"raw_sha256=sha256:{observed_raw_hash}")
    print(f"processed_records={record_count}")
    print(f"contract_id={persisted_contract.get('contract_id')}")
    print("training_launched=false")


if __name__ == "__main__":
    main()
