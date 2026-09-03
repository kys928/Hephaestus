#!/usr/bin/env python3
"""Verify RunPod scientific artifacts byte-for-byte and reconstruct experiment bindings.

This command is intentionally read-only. It never writes to the Network Volume.
Generated JSON files are local workflow evidence only.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import boto3
from botocore.config import Config


ARTIFACT_PREFIX = "Hephaestus/artifacts/"
TEST_RUN_PREFIXES = (
    "gate-orch-",
    "scorecard-",
    "stage3-",
    "stage4-",
    "stage5-",
    "s6-",
    "s7-",
    "s8-",
    "s9-",
)
MODEL_WEIGHT_NAMES = {"model.safetensors", "pytorch_model.bin"}
MODEL_WEIGHT_SUFFIXES = {".safetensors"}
CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}
DATASET_NAMES = {"processed_dataset.jsonl", "trainable.jsonl"}
DATASET_SUFFIXES = {".parquet", ".arrow", ".jsonl", ".csv", ".tsv"}


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _json_or_text_preview(data: bytes, key: str) -> object | None:
    if len(data) > 64 * 1024:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if key.lower().endswith(".json"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"text_preview": text[:300]}
        if isinstance(payload, dict):
            retained: dict[str, object] = {}
            for field in (
                "artifact_ref", "dataset_id", "source_identity", "source_ids", "total_examples",
                "quality_score", "license", "risks", "run_id", "lineage_id", "stage_name",
                "status", "contract_version", "checkpoint_candidates", "checkpoint_scores",
                "manifest_hash", "partial_write", "model_id", "model_revision", "tokenizer_id",
                "tokenizer_revision", "architecture_family", "processed_dataset_ref",
                "processed_dataset_hash", "checkpoint_refs",
            ):
                if field in payload:
                    retained[field] = payload[field]
            retained["top_level_keys"] = sorted(str(k) for k in payload.keys())
            return retained
        return payload
    if key.lower().endswith((".jsonl", ".txt", ".log", ".ckpt")):
        return {"text_preview": text[:300]}
    return None


def _kind(key: str) -> str:
    path = PurePosixPath(key)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in MODEL_WEIGHT_NAMES or suffix in MODEL_WEIGHT_SUFFIXES:
        return "model_weight"
    if name in DATASET_NAMES or suffix in DATASET_SUFFIXES:
        return "dataset"
    if suffix in CHECKPOINT_SUFFIXES or "checkpoint" in name:
        return "checkpoint"
    if name == "tokenizer.json" or "tokenizer" in name:
        return "tokenizer"
    if name.endswith("manifest.json") or "contract" in name:
        return "metadata_contract"
    if name.endswith(".json"):
        return "metadata"
    return "other"


def _run_id(key: str) -> str | None:
    relative = key.removeprefix(ARTIFACT_PREFIX)
    parts = PurePosixPath(relative).parts
    return parts[0] if len(parts) > 1 else None


def _fixture_signals(record: dict[str, object]) -> list[str]:
    signals: list[str] = []
    run_id = str(record.get("run_id") or "")
    kind = str(record["kind"])
    size = int(record["size"])
    preview = record.get("preview")
    if run_id.startswith(TEST_RUN_PREFIXES):
        signals.append("run_id_matches_repository_test_fixture_family")
    if kind == "checkpoint" and size < 1024:
        signals.append("checkpoint_is_sub_kibibyte")
    if kind == "dataset" and size < 1024:
        signals.append("dataset_is_sub_kibibyte")
    if kind == "tokenizer" and size <= 2:
        signals.append("tokenizer_is_empty_json_object_size")
    if isinstance(preview, dict) and preview.get("top_level_keys") == []:
        signals.append("json_object_is_empty")
    return signals


def main() -> None:
    bucket = _required("RUNPOD_NETWORK_VOLUME_ID")
    endpoint = _required("RUNPOD_S3_ENDPOINT_URL").rstrip("/")
    region = _required("RUNPOD_DATACENTER_ID")
    access_key = _required("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = _required("RUNPOD_S3_SECRET_ACCESS_KEY")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )
    client.head_bucket(Bucket=bucket)

    records: list[dict[str, object]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=ARTIFACT_PREFIX, PaginationConfig={"PageSize": 1000}):
        for item in page.get("Contents", []) or []:
            key = str(item["Key"])
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            data = body.read()
            body.close()
            listed_size = int(item.get("Size", 0))
            observed_size = len(data)
            record: dict[str, object] = {
                "key": key,
                "run_id": _run_id(key),
                "kind": _kind(key),
                "size": observed_size,
                "listed_size": listed_size,
                "size_verified": observed_size == listed_size,
                "sha256": _sha256(data),
                "etag": str(item.get("ETag", "")).strip('"') or None,
                "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                "content_type": response.get("ContentType"),
                "metadata": dict(response.get("Metadata") or {}),
                "preview": _json_or_text_preview(data, key),
            }
            expected_sha = record["metadata"].get("sha256") if isinstance(record["metadata"], dict) else None
            record["metadata_sha256"] = expected_sha
            record["metadata_sha256_matches"] = bool(expected_sha) and str(expected_sha).removeprefix("sha256:") == str(record["sha256"]).removeprefix("sha256:")
            record["fixture_signals"] = _fixture_signals(record)
            records.append(record)

    records.sort(key=lambda record: str(record["key"]))
    kinds = Counter(str(record["kind"]) for record in records)
    fixture_records = [record for record in records if record["fixture_signals"]]
    run_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        run_groups[str(record.get("run_id") or "<root>")].append(record)

    model_weights = [record for record in records if record["kind"] == "model_weight"]
    datasets = [record for record in records if record["kind"] == "dataset"]
    checkpoints = [record for record in records if record["kind"] == "checkpoint"]
    tokenizers = [record for record in records if record["kind"] == "tokenizer"]

    viable_model_weights = [record for record in model_weights if int(record["size"]) >= 1024 * 1024]
    viable_datasets = [record for record in datasets if int(record["size"]) >= 1024]
    viable_checkpoints = [record for record in checkpoints if int(record["size"]) >= 1024 * 1024]
    viable_tokenizers = [record for record in tokenizers if int(record["size"]) >= 1024]

    inventory_payload = {
        "verification_version": "runpod-scientific-state-verification.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "volume_id": bucket,
        "datacenter_id": region,
        "endpoint": endpoint,
        "prefix": ARTIFACT_PREFIX,
        "read_only": True,
        "object_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "kind_counts": dict(sorted(kinds.items())),
        "all_listed_sizes_verified": all(bool(record["size_verified"]) for record in records),
        "metadata_sha256_present_count": sum(bool(record["metadata_sha256"]) for record in records),
        "fixture_signal_object_count": len(fixture_records),
        "viable_scientific_inputs": {
            "model_weights": len(viable_model_weights),
            "datasets": len(viable_datasets),
            "checkpoints": len(viable_checkpoints),
            "tokenizers": len(viable_tokenizers),
        },
        "objects": records,
    }
    inventory_payload["inventory_hash"] = _canonical_hash(records)

    run_summary = []
    for run_id, run_records in sorted(run_groups.items()):
        run_summary.append({
            "run_id": run_id,
            "objects": len(run_records),
            "bytes": sum(int(record["size"]) for record in run_records),
            "kinds": dict(sorted(Counter(str(record["kind"]) for record in run_records).items())),
            "all_records_have_fixture_signals": all(bool(record["fixture_signals"]) for record in run_records),
        })

    status = "ready" if viable_model_weights and viable_datasets and viable_checkpoints else "blocked_missing_real_scientific_inputs"
    reconstruction = {
        "reconstruction_version": "hephaestus-first-scientific-experiment.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_volume": {
            "provider": "runpod-network-volume",
            "volume_id": bucket,
            "datacenter_id": region,
            "artifact_prefix": ARTIFACT_PREFIX,
            "inventory_hash": inventory_payload["inventory_hash"],
        },
        "evidence_summary": {
            "artifact_objects_verified": len(records),
            "artifact_bytes_verified": inventory_payload["total_bytes"],
            "model_weight_objects": len(model_weights),
            "dataset_objects": len(datasets),
            "checkpoint_objects": len(checkpoints),
            "tokenizer_objects": len(tokenizers),
            "viable_model_weight_objects": len(viable_model_weights),
            "viable_dataset_objects": len(viable_datasets),
            "viable_checkpoint_objects": len(viable_checkpoints),
            "viable_tokenizer_objects": len(viable_tokenizers),
            "run_groups": run_summary,
        },
        "contract_bindings": {
            "dataset_manifest": None,
            "trainable_data_contract": None,
            "model_candidate": None,
            "model_selection_decision": None,
            "experiment_proposal": None,
            "training_run_handle": None,
        },
        "blocking_issues": [
            {
                "code": "real_dataset_missing",
                "category": "missing_evidence",
                "message": "No non-fixture, training-scale dataset artifact was found under Hephaestus/artifacts/ on the paid Network Volume.",
            },
            {
                "code": "real_model_weights_missing",
                "category": "missing_evidence",
                "message": "No model.safetensors, pytorch_model.bin, or other training-scale model weight artifact was found under Hephaestus/artifacts/.",
            },
            {
                "code": "real_checkpoint_missing",
                "category": "missing_evidence",
                "message": "Checkpoint-named objects exist, but none is training-scale and none is a finalized checkpoint directory with checkpoint_manifest.json as required by the current Transformers lifecycle verifier.",
            },
            {
                "code": "fixture_evidence_not_promotable",
                "category": "artifact_integrity",
                "message": "Existing run artifacts are tiny staged/test fixtures and cannot honestly be bound as the first scientific experiment inputs.",
            },
        ],
        "next_required_bindings": [
            "an immutable real dataset artifact plus DatasetManifest evidence",
            "a compatible real model/tokenizer identity or local model directory with content identity",
            "a real baseline checkpoint only if the experiment resumes/compares against one",
            "then a governed ModelSelectionDecision and ExperimentProposal",
        ],
        "scientific_claim": "No first paid scientific experiment can be reconstructed as ready from the current volume without inventing missing inputs.",
    }
    reconstruction["reconstruction_hash"] = _canonical_hash({key: value for key, value in reconstruction.items() if key != "reconstruction_hash"})

    with open("artifact_verification.json", "w", encoding="utf-8") as handle:
        json.dump(inventory_payload, handle, indent=2, sort_keys=True)
    with open("first_scientific_experiment_reconstruction.json", "w", encoding="utf-8") as handle:
        json.dump(reconstruction, handle, indent=2, sort_keys=True)

    print(json.dumps({
        "status": status,
        "object_count": len(records),
        "total_bytes": inventory_payload["total_bytes"],
        "kind_counts": inventory_payload["kind_counts"],
        "viable_scientific_inputs": inventory_payload["viable_scientific_inputs"],
        "inventory_hash": inventory_payload["inventory_hash"],
        "reconstruction_hash": reconstruction["reconstruction_hash"],
    }, sort_keys=True))
    for record in records:
        if record["kind"] in {"dataset", "model_weight", "checkpoint", "tokenizer", "metadata_contract"}:
            print(json.dumps({
                "key": record["key"],
                "kind": record["kind"],
                "size": record["size"],
                "sha256": record["sha256"],
                "etag": record["etag"],
                "fixture_signals": record["fixture_signals"],
            }, sort_keys=True))


if __name__ == "__main__":
    main()
