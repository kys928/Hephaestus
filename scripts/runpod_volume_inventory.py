#!/usr/bin/env python3
"""Read-only inventory of scientific artifacts on the configured RunPod Network Volume."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath

import boto3
from botocore.config import Config


JUNK_PREFIXES = (
    ".cache/pip/",
    ".cache/uv/",
    ".cache/torch/",
    ".cache/triton/",
    ".local/",
    ".npm/",
    ".cargo/",
    ".rustup/",
    ".conda/",
    "tmp/",
    "temp/",
    "var/cache/",
    "usr/",
    "opt/conda/",
)
JUNK_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "site-packages", "node_modules"}
DATA_EXTENSIONS = {".jsonl", ".parquet", ".arrow", ".csv", ".tsv", ".txt", ".json", ".bin"}
MODEL_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
}
CHECKPOINT_FILES = {
    "checkpoint_manifest.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "scaler.pt",
}
METADATA_NAMES = {
    "dataset_manifest.json",
    "trainable_data_contract.json",
    "preprocessing_report.json",
    "processing_evidence.json",
    "prepared_job.json",
    "normalized_training_config.json",
    "resource_estimate.json",
    "checkpoint_record.json",
    "runtime_result.json",
    "training_run_handle.json",
}


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def _junk_reason(key: str) -> str | None:
    lower = key.lower().lstrip("/")
    if lower.startswith(JUNK_PREFIXES):
        return "environment_cache"
    parts = {part.lower() for part in PurePosixPath(lower).parts}
    if parts & JUNK_PARTS:
        return "environment_tree"
    if lower.endswith((".pyc", ".pyo", ".log", ".tmp", ".lock")):
        return "runtime_noise"
    return None


def _classify(key: str) -> str | None:
    lower = key.lower()
    name = PurePosixPath(lower).name
    parts = set(PurePosixPath(lower).parts)

    checkpoint_context = any("checkpoint" in part or part.startswith("ckpt") for part in parts)
    model_context = any(part in {"model", "models", "weights", "tokenizer", "tokenizers"} for part in parts)
    data_context = any(part in {"data", "dataset", "datasets", "corpus", "corpora", "train", "training_data", "eval", "validation"} for part in parts)

    if name in CHECKPOINT_FILES or checkpoint_context:
        return "checkpoint"
    if name in MODEL_FILES or model_context or name.endswith(".safetensors"):
        return "model"
    if name in METADATA_NAMES or "manifest" in name or "contract" in name:
        return "scientific_metadata"
    suffix = PurePosixPath(lower).suffix
    if data_context and suffix in DATA_EXTENSIONS:
        return "dataset"
    if any(token in lower for token in ("dataset", "corpus", "trainable.jsonl", "preprocessed")) and suffix in DATA_EXTENSIONS:
        return "dataset"
    return None


def _safe_json_preview(client, bucket: str, key: str, size: int) -> dict[str, object] | None:
    if size <= 0 or size > 2 * 1024 * 1024 or not key.lower().endswith(".json"):
        return None
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        raw = body.read()
        body.close()
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return {"json_type": type(payload).__name__}
    useful = {}
    for field in (
        "manifest_id", "run_id", "lineage_id", "stage_name", "contract_id", "dataset_id",
        "model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "architecture",
        "architecture_family", "status", "backend_id", "experiment_id", "manifest_hash",
        "partial_write", "row_count", "total_examples", "content_hash", "processed_dataset_ref",
        "processed_dataset_hash", "checkpoint_ref", "checkpoint_refs",
    ):
        if field in payload:
            useful[field] = payload[field]
    useful["top_level_keys"] = sorted(str(k) for k in payload.keys())[:80]
    return useful


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

    paginator = client.get_paginator("list_objects_v2")
    all_objects: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    ignored = Counter()
    categories = Counter()
    prefix_bytes: dict[str, int] = defaultdict(int)
    prefix_counts = Counter()

    for page in paginator.paginate(Bucket=bucket, PaginationConfig={"PageSize": 1000}):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            size = int(item.get("Size", 0))
            last_modified = item.get("LastModified")
            timestamp = last_modified.isoformat() if last_modified is not None else None
            root_prefix = key.split("/", 1)[0] if "/" in key else "<root>"
            prefix_bytes[root_prefix] += size
            prefix_counts[root_prefix] += 1
            row = {"key": key, "size": size, "last_modified": timestamp, "etag": str(item.get("ETag", "")).strip('"') or None}
            all_objects.append(row)

            reason = _junk_reason(key)
            if reason:
                ignored[reason] += 1
                continue
            category = _classify(key)
            if category is None:
                ignored["unclassified_non_scientific"] += 1
                continue

            categories[category] += 1
            try:
                head = client.head_object(Bucket=bucket, Key=key)
            except Exception as exc:
                head = {"head_error": type(exc).__name__}
            candidate = {
                **row,
                "category": category,
                "content_type": head.get("ContentType"),
                "metadata": head.get("Metadata") if isinstance(head.get("Metadata"), dict) else {},
                "storage_class": head.get("StorageClass"),
                "checksum_sha256": head.get("ChecksumSHA256"),
                "json_preview": _safe_json_preview(client, bucket, key, size),
            }
            candidates.append(candidate)

    all_objects.sort(key=lambda x: str(x["key"]))
    candidates.sort(key=lambda x: (str(x["category"]), str(x["key"])))
    summary = {
        "inventory_version": "runpod-volume-inventory.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "volume_id": bucket,
        "datacenter_id": region,
        "endpoint": endpoint,
        "read_only": True,
        "total_objects": len(all_objects),
        "total_bytes": sum(int(x["size"]) for x in all_objects),
        "candidate_objects": len(candidates),
        "candidate_bytes": sum(int(x["size"]) for x in candidates),
        "category_counts": dict(sorted(categories.items())),
        "ignored_counts": dict(sorted(ignored.items())),
        "top_level_prefixes": [
            {"prefix": prefix, "objects": prefix_counts[prefix], "bytes": prefix_bytes[prefix]}
            for prefix in sorted(prefix_counts, key=lambda p: (-prefix_bytes[p], p))
        ],
    }
    output = {"summary": summary, "candidates": candidates}
    with open("volume_inventory.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    with open("full_object_index.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "objects": all_objects}, fh, indent=2, sort_keys=True)
    with open("candidate_paths.txt", "w", encoding="utf-8") as fh:
        for item in candidates:
            fh.write(f"{item['category']}\t{item['size']}\t{item['key']}\n")

    print(json.dumps(summary, sort_keys=True))
    for item in candidates[:250]:
        print(json.dumps({k: item[k] for k in ("category", "key", "size", "last_modified", "etag", "json_preview")}, sort_keys=True))


if __name__ == "__main__":
    main()
