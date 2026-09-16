#!/usr/bin/env python3
"""Read-only inventory of the Hephaestus RunPod Network Volume through S3.

This script never deletes or mutates objects. It exists to give a human/agent an
actual byte-level inventory before any maintenance pod is allowed to delete
specific paths.
"""
from __future__ import annotations

import collections
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import launch_first_bounded_scientific_training as base

VOLUME_ID = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "cviwpryzao").strip() or "cviwpryzao"
ACTIVE_ELASTICITY_RUN = "adaptation-elasticity-v1-34961824753"
OUT_JSON = Path("network_volume_s3_inventory.json")
OUT_TSV = Path("network_volume_s3_largest.tsv")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "astimezone"):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _protected_reason(key: str) -> str | None:
    active = f"hephaestus/scientific/v1/adaptation_elasticity/{ACTIVE_ELASTICITY_RUN}/"
    active_exec = f"hephaestus/scientific/v1/executions/{ACTIVE_ELASTICITY_RUN}/"
    if key.startswith(active) or key.startswith(active_exec):
        return "active_adaptation_elasticity"
    if "/adapters/" in key:
        return "scientific_adapter"
    if "/checkpoint" in key.casefold() or "/checkpoints/" in key.casefold():
        return "checkpoint_or_rollback_material"
    if key.startswith("hephaestus/scientific/v1/"):
        return "scientific_evidence_or_manifest"
    return None


def _category(key: str) -> str:
    lower = key.casefold()
    if "/adaptation_elasticity/" in lower:
        return "adaptation_elasticity"
    if "/cognitive_topology/" in lower:
        return "cognitive_topology"
    if "/model_admission/" in lower:
        return "model_admission"
    if "/executions/" in lower:
        return "executions"
    if "/maintenance/" in lower:
        return "maintenance"
    if "checkpoint" in lower:
        return "checkpoints"
    if "/adapters/" in lower:
        return "adapters"
    if "cache" in lower or "/snapshots/" in lower or "/models--" in lower:
        return "cache_or_snapshot"
    if lower.startswith("hephaestus/scientific/"):
        return "other_scientific"
    if lower.startswith("hephaestus/"):
        return "hephaestus_other"
    return "other"


def _prefixes(key: str, max_depth: int = 7) -> list[tuple[int, str]]:
    parts = [part for part in key.split("/") if part]
    return [(depth, "/".join(parts[:depth]) + "/") for depth in range(1, min(max_depth, len(parts)) + 1)]


def main() -> int:
    client = base.s3_client()
    client.head_bucket(Bucket=VOLUME_ID)

    objects: list[dict[str, Any]] = []
    continuation: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": VOLUME_ID, "MaxKeys": 1000}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        page = client.list_objects_v2(**kwargs)
        for row in page.get("Contents", []) or []:
            key = str(row.get("Key") or "")
            if not key:
                continue
            objects.append(
                {
                    "key": key,
                    "bytes": int(row.get("Size") or 0),
                    "last_modified": _iso(row.get("LastModified")),
                    "etag": str(row.get("ETag") or "").strip('"') or None,
                    "storage_class": row.get("StorageClass"),
                    "category": _category(key),
                    "protected_reason": _protected_reason(key),
                }
            )
        if not page.get("IsTruncated"):
            break
        continuation = page.get("NextContinuationToken")
        if not continuation:
            raise RuntimeError("S3 inventory pagination truncated without continuation token")

    prefix_bytes: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    prefix_count: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    category_bytes: collections.Counter[str] = collections.Counter()
    category_count: collections.Counter[str] = collections.Counter()
    suffix_bytes: collections.Counter[str] = collections.Counter()
    suffix_count: collections.Counter[str] = collections.Counter()

    for row in objects:
        key = row["key"]
        size = int(row["bytes"])
        category_bytes[row["category"]] += size
        category_count[row["category"]] += 1
        suffix = Path(key).suffix.casefold() or "<none>"
        suffix_bytes[suffix] += size
        suffix_count[suffix] += 1
        for depth, prefix in _prefixes(key):
            prefix_bytes[depth][prefix] += size
            prefix_count[depth][prefix] += 1

    largest = sorted(objects, key=lambda row: int(row["bytes"]), reverse=True)[:300]
    prefix_usage: dict[str, list[dict[str, Any]]] = {}
    for depth in sorted(prefix_bytes):
        rows = []
        for prefix, size in prefix_bytes[depth].most_common(150):
            rows.append({"prefix": prefix, "bytes": int(size), "object_count": int(prefix_count[depth][prefix])})
        prefix_usage[str(depth)] = rows

    total_bytes = sum(int(row["bytes"]) for row in objects)
    protected_bytes = sum(int(row["bytes"]) for row in objects if row["protected_reason"])
    unprotected_bytes = total_bytes - protected_bytes
    payload = {
        "inventory_version": "network-volume-s3-inventory.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "volume_id": VOLUME_ID,
        "active_protected_run": ACTIVE_ELASTICITY_RUN,
        "read_only": True,
        "object_count": len(objects),
        "total_bytes": total_bytes,
        "protected_bytes": protected_bytes,
        "unprotected_bytes": unprotected_bytes,
        "category_usage": [
            {"category": category, "bytes": int(size), "object_count": int(category_count[category])}
            for category, size in category_bytes.most_common()
        ],
        "suffix_usage": [
            {"suffix": suffix, "bytes": int(size), "object_count": int(suffix_count[suffix])}
            for suffix, size in suffix_bytes.most_common(100)
        ],
        "prefix_usage": prefix_usage,
        "largest_objects": largest,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_TSV.write_text(
        "bytes\tcategory\tprotected_reason\tlast_modified\tkey\n"
        + "".join(
            f"{row['bytes']}\t{row['category']}\t{row['protected_reason'] or ''}\t{row['last_modified'] or ''}\t{row['key']}\n"
            for row in largest
        ),
        encoding="utf-8",
    )
    print(
        "NETWORK_VOLUME_S3_INVENTORY_JSON "
        + json.dumps(
            {
                "object_count": len(objects),
                "total_bytes": total_bytes,
                "protected_bytes": protected_bytes,
                "unprotected_bytes": unprotected_bytes,
                "top_categories": payload["category_usage"][:10],
                "top_depth4_prefixes": prefix_usage.get("4", [])[:20],
                "largest_objects": largest[:30],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
