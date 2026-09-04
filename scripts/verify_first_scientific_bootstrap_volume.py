#!/usr/bin/env python3
"""Independently verify the first scientific bootstrap on the RunPod Network Volume.

This command is read-only. It starts from the immutable bootstrap bundle, reads
referenced objects back from the paid Network Volume, recomputes SHA-256, rebuilds
model/tokenizer directory identities, and parses the four typed contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import PurePosixPath
from typing import Any

import boto3
from botocore.config import Config

from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.discovery_contract import ModelSelectionDecision
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.schemas.trainable_data_contract import TrainableDataContract

PREFIX = "hephaestus/scientific/v1"
EXPECTED_BUNDLE = "sha256:6774e92d2b595353a18211ffa772fb82b362d462ee2e3c26144f705d26525436"
EXPECTED_RAW = "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
EXPECTED_PROCESSED = "sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c"
EXPECTED_TOKENIZER = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
EXPECTED_MODEL = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
EXPECTED_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EXPECTED_DATASET_MANIFEST_ID = "manifest-first-scientific-bootstrap-001"
EXPECTED_TRAINABLE_DATA_ID = "trainable-data-first-scientific-bootstrap-001"
EXPECTED_MODEL_SELECTION_ID = "model-selection-model-search-b6744896f7e3a16c"
EXPECTED_EXPERIMENT_ID = "experiment-60bff7cb4f478f91"
EXPECTED_PARAMETER_COUNT = 1_874_688
EXPECTED_RAW_BYTES = 6_357_543


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def _digest(value: str) -> str:
    digest = value.removeprefix("sha256:").lower()
    if len(digest) != 64:
        raise ValueError(f"invalid sha256 identity: {value!r}")
    int(digest, 16)
    return digest


def _object_key(artifact_ref: str) -> str:
    digest = _digest(artifact_ref)
    return f"{PREFIX}/objects/sha256/{digest[:2]}/{digest}"


def _key_from_s3(path: str, bucket: str) -> str:
    prefix = f"s3://{bucket}/"
    if not path.startswith(prefix):
        raise ValueError(f"storage path does not target expected volume: {path}")
    key = path.removeprefix(prefix)
    if not key.startswith(f"{PREFIX}/"):
        raise ValueError(f"storage path escapes scientific prefix: {path}")
    return key


def _read_verified(
    client: Any,
    bucket: str,
    key: str,
    expected_hash: str,
    *,
    expected_size: int | None = None,
) -> bytes:
    expected = _digest(expected_hash)
    head = client.head_object(Bucket=bucket, Key=key)
    listed_size = int(head.get("ContentLength", -1))
    if expected_size is not None and listed_size != expected_size:
        raise RuntimeError(f"size mismatch for {key}: {listed_size}!={expected_size}")
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    observed = 0
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            hasher.update(chunk)
    finally:
        body.close()
    if observed != listed_size:
        raise RuntimeError(f"GET byte count disagrees with HeadObject for {key}")
    if hasher.hexdigest() != expected:
        raise RuntimeError(f"sha256 mismatch for {key}")
    return b"".join(chunks)


def _json_verified(client: Any, bucket: str, record: dict[str, object]) -> dict[str, object]:
    artifact_ref = str(record["artifact_ref"])
    content_hash = str(record.get("content_hash") or artifact_ref)
    if artifact_ref != content_hash:
        raise RuntimeError("evidence record artifact_ref/content_hash disagreement")
    key = _key_from_s3(str(record["storage_path"]), bucket)
    if key != _object_key(artifact_ref):
        raise RuntimeError("evidence storage path is not the content-addressed key")
    raw = _read_verified(
        client,
        bucket,
        key,
        artifact_ref,
        expected_size=int(record["byte_size"]) if record.get("byte_size") is not None else None,
    )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {key}")
    return {str(k): v for k, v in payload.items()}


def _directory_identity(component_hashes: dict[str, str]) -> str:
    encoded = json.dumps(
        {path: _digest(value) for path, value in component_hashes.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _verify_directory(
    client: Any,
    bucket: str,
    manifest: dict[str, object],
    expected_identity: str,
) -> dict[str, object]:
    if str(manifest.get("directory_content_identity")) != expected_identity:
        raise RuntimeError("directory manifest identity does not match bootstrap bundle")
    if str(manifest.get("volume_id")) != bucket:
        raise RuntimeError("directory manifest references another volume")
    raw_components = manifest.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        raise RuntimeError("directory manifest has no components")

    component_hashes: dict[str, str] = {}
    verified_components: list[dict[str, object]] = []
    for relative_path, raw_record in sorted(raw_components.items()):
        if not isinstance(raw_record, dict):
            raise RuntimeError("directory component evidence is malformed")
        expected_hash = str(raw_record["sha256"])
        size = int(raw_record["byte_size"])
        materialized_key = _key_from_s3(str(raw_record["materialized_storage_path"]), bucket)
        content_key = _key_from_s3(str(raw_record["content_storage_path"]), bucket)
        content_ref = str(raw_record["content_artifact_ref"])
        if content_key != _object_key(content_ref) or content_ref != expected_hash:
            raise RuntimeError(f"content-addressed component binding is invalid: {relative_path}")
        _read_verified(client, bucket, materialized_key, expected_hash, expected_size=size)
        _read_verified(client, bucket, content_key, expected_hash, expected_size=size)
        component_hashes[str(relative_path)] = expected_hash
        verified_components.append(
            {
                "path": str(relative_path),
                "sha256": expected_hash,
                "byte_size": size,
                "materialized_key": materialized_key,
                "content_key": content_key,
            }
        )

    rebuilt = _directory_identity(component_hashes)
    if rebuilt != expected_identity:
        raise RuntimeError(f"rebuilt directory identity mismatch: {rebuilt}!={expected_identity}")
    return {
        "directory_content_identity": rebuilt,
        "component_count": len(verified_components),
        "components": verified_components,
    }


def main() -> None:
    bucket = _required("RUNPOD_NETWORK_VOLUME_ID")
    endpoint = _required("RUNPOD_S3_ENDPOINT_URL").rstrip("/")
    region = _required("RUNPOD_DATACENTER_ID")
    access_key = _required("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = _required("RUNPOD_S3_SECRET_ACCESS_KEY")
    if bucket != "cviwpryzao" or region != "EU-CZ-1":
        raise SystemExit("unexpected RunPod scientific volume contract")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )
    client.head_bucket(Bucket=bucket)

    bundle_bytes = _read_verified(client, bucket, _object_key(EXPECTED_BUNDLE), EXPECTED_BUNDLE)
    bundle = json.loads(bundle_bytes.decode("utf-8"))
    if not isinstance(bundle, dict):
        raise RuntimeError("bootstrap bundle is not a JSON object")

    dataset = dict(bundle.get("dataset") or {})
    model = dict(bundle.get("model") or {})
    tokenizer = dict(bundle.get("tokenizer") or {})
    typed_chain = dict(bundle.get("typed_chain") or {})
    if dataset.get("dataset_id") != "Salesforce/wikitext":
        raise RuntimeError("unexpected dataset identity in bundle")
    if dataset.get("resolved_revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError("dataset revision drifted in staged bundle")
    if dataset.get("raw_content_hash") != EXPECTED_RAW:
        raise RuntimeError("raw dataset hash drifted in staged bundle")
    if dataset.get("processed_content_hash") != EXPECTED_PROCESSED:
        raise RuntimeError("processed dataset hash drifted in staged bundle")
    if tokenizer.get("directory_content_identity") != EXPECTED_TOKENIZER:
        raise RuntimeError("tokenizer identity drifted in staged bundle")
    if model.get("directory_content_identity") != EXPECTED_MODEL:
        raise RuntimeError("model identity drifted in staged bundle")
    if int(model.get("parameter_count", 0)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("model parameter count drifted in staged bundle")

    _read_verified(client, bucket, _object_key(EXPECTED_RAW), EXPECTED_RAW, expected_size=EXPECTED_RAW_BYTES)
    processed_record = dict(dict(bundle["evidence"])["processed_dataset"])
    _read_verified(
        client,
        bucket,
        _object_key(EXPECTED_PROCESSED),
        EXPECTED_PROCESSED,
        expected_size=int(processed_record["byte_size"]),
    )

    evidence = dict(bundle.get("evidence") or {})
    verified_evidence: dict[str, dict[str, object]] = {}
    payloads: dict[str, dict[str, object]] = {}
    for name, raw_record in sorted(evidence.items()):
        if not isinstance(raw_record, dict) or not raw_record.get("artifact_ref"):
            continue
        payload = _json_verified(client, bucket, raw_record) if name != "processed_dataset" else {}
        verified_evidence[str(name)] = {
            "artifact_ref": raw_record["artifact_ref"],
            "byte_size": raw_record.get("byte_size"),
            "storage_path": raw_record.get("storage_path"),
        }
        if payload:
            payloads[str(name)] = payload

    required_names = {
        "dataset_manifest",
        "trainable_data_contract",
        "model_selection_decision",
        "experiment_proposal",
        "model_directory_manifest",
        "tokenizer_directory_manifest",
    }
    missing = required_names - payloads.keys()
    if missing:
        raise RuntimeError(f"bootstrap bundle is missing required evidence payloads: {sorted(missing)}")

    dataset_manifest = DatasetManifest.from_dict(payloads["dataset_manifest"])
    trainable_data = TrainableDataContract.from_dict(payloads["trainable_data_contract"])
    model_selection = ModelSelectionDecision.from_dict(payloads["model_selection_decision"])
    experiment = ExperimentProposal.from_dict(payloads["experiment_proposal"])

    if dataset_manifest.manifest_id != EXPECTED_DATASET_MANIFEST_ID:
        raise RuntimeError("DatasetManifest ID mismatch")
    if trainable_data.contract_id != EXPECTED_TRAINABLE_DATA_ID:
        raise RuntimeError("TrainableDataContract ID mismatch")
    if trainable_data.manifest_id != dataset_manifest.manifest_id:
        raise RuntimeError("TrainableDataContract does not bind DatasetManifest")
    if model_selection.decision_id != EXPECTED_MODEL_SELECTION_ID or model_selection.status != "selected":
        raise RuntimeError("ModelSelectionDecision is not the expected selected decision")
    if experiment.experiment_id != EXPECTED_EXPERIMENT_ID:
        raise RuntimeError("ExperimentProposal ID mismatch")
    if experiment.model_selection_id != model_selection.decision_id:
        raise RuntimeError("ExperimentProposal does not bind ModelSelectionDecision")
    if typed_chain != {
        "dataset_manifest_id": EXPECTED_DATASET_MANIFEST_ID,
        "trainable_data_contract_id": EXPECTED_TRAINABLE_DATA_ID,
        "model_selection_decision_id": EXPECTED_MODEL_SELECTION_ID,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
    }:
        raise RuntimeError("bundle typed_chain does not match parsed contracts")

    tokenizer_verification = _verify_directory(
        client, bucket, payloads["tokenizer_directory_manifest"], EXPECTED_TOKENIZER
    )
    model_verification = _verify_directory(
        client, bucket, payloads["model_directory_manifest"], EXPECTED_MODEL
    )

    launch_boundary = dict(bundle.get("launch_boundary") or {})
    if launch_boundary.get("training_launched") is not False or launch_boundary.get("runpod_pod_created") is not False:
        raise RuntimeError("bootstrap unexpectedly crossed the training launch boundary")

    result = {
        "verification_version": "first-scientific-volume-verification.v1",
        "status": "verified",
        "volume_id": bucket,
        "datacenter_id": region,
        "bundle_artifact_ref": EXPECTED_BUNDLE,
        "dataset": {
            "dataset_id": "Salesforce/wikitext",
            "revision": EXPECTED_DATASET_REVISION,
            "raw_sha256": EXPECTED_RAW,
            "raw_bytes": EXPECTED_RAW_BYTES,
            "processed_sha256": EXPECTED_PROCESSED,
            "processed_bytes": int(processed_record["byte_size"]),
        },
        "tokenizer": tokenizer_verification,
        "model": {
            **model_verification,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "forward_smoke": model.get("forward_smoke"),
        },
        "typed_chain": typed_chain,
        "typed_contracts_parsed": True,
        "evidence_object_count_verified": len(verified_evidence),
        "launch_boundary": launch_boundary,
    }
    with open("first_scientific_volume_verification.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
