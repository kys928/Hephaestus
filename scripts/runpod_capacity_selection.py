#!/usr/bin/env python3
"""Query RunPod's hardware catalog for single-GPU Secure Cloud capacity in the bound datacenter.

This is execution routing evidence, not a scientific experiment variable. The
caller keeps model/data/recipe/evaluation contracts fixed and uses this helper
only to avoid blind create-Pod requests when a GPU SKU is temporarily exhausted.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

GRAPHQL_URL = "https://api.runpod.io/graphql"
DATACENTER = "EU-CZ-1"
MIN_MEMORY_GB = 16

# Prefer the hardware used by the first training run, then inexpensive single-GPU
# Secure Cloud alternatives. IDs discovered live take precedence over these
# display-name hints; no unverified ID is invented.
PREFERENCE = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA L4",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
)


def _required_api_key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        raise RuntimeError("RUNPOD_API_KEY is required for capacity discovery")
    return value


def _query() -> list[dict[str, Any]]:
    query = f'''query {{
      gpuTypes {{
        id
        displayName
        memoryInGb
        secureCloud
        lowestPrice(input: {{gpuCount: 1, secureCloud: true, dataCenterId: "{DATACENTER}"}}) {{
          gpuTypeId
          stockStatus
          availableGpuCounts
          uninterruptablePrice
          totalCount
          rentedCount
          maxUnreservedGpuCount
        }}
      }}
    }}'''
    # RunPod's documented GraphQL management path authenticates catalog queries
    # with the api_key query parameter. Do not persist or print the expanded URL.
    url = GRAPHQL_URL + "?api_key=" + urllib.parse.quote(_required_api_key(), safe="")
    request = urllib.request.Request(
        url,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"RunPod capacity query HTTP {exc.code}: {body}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RunPod capacity query returned a non-object payload")
    if payload.get("errors"):
        raise RuntimeError(f"RunPod capacity query GraphQL errors: {payload['errors']}")
    data = payload.get("data")
    rows = data.get("gpuTypes") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("RunPod capacity query did not return gpuTypes")
    return [dict(row) for row in rows if isinstance(row, dict)]


def discover_available_gpu_ids() -> tuple[list[str], dict[str, object]]:
    rows = _query()
    candidates: list[dict[str, object]] = []
    for row in rows:
        gpu_id = str(row.get("id") or "").strip()
        display = str(row.get("displayName") or gpu_id).strip()
        try:
            memory = int(row.get("memoryInGb") or 0)
        except (TypeError, ValueError):
            memory = 0
        lowest = row.get("lowestPrice")
        if not gpu_id or memory < MIN_MEMORY_GB or row.get("secureCloud") is not True or not isinstance(lowest, dict):
            continue
        counts = lowest.get("availableGpuCounts")
        count_values = [int(value) for value in counts if isinstance(value, (int, float))] if isinstance(counts, list) else []
        available = 1 in count_values or int(lowest.get("maxUnreservedGpuCount") or 0) >= 1
        if not available:
            continue
        candidates.append(
            {
                "id": gpu_id,
                "display_name": display,
                "memory_gb": memory,
                "stock_status": lowest.get("stockStatus"),
                "available_gpu_counts": count_values,
                "max_unreserved_gpu_count": lowest.get("maxUnreservedGpuCount"),
                "uninterruptable_price": lowest.get("uninterruptablePrice"),
            }
        )

    preference = {name: index for index, name in enumerate(PREFERENCE)}
    candidates.sort(
        key=lambda item: (
            preference.get(str(item["id"]), preference.get(str(item["display_name"]), len(preference) + 1)),
            float(item["uninterruptable_price"] or 9999.0),
            str(item["display_name"]),
        )
    )
    ids = [str(item["id"]) for item in candidates]
    evidence = {
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "graphql_endpoint": GRAPHQL_URL,
        "authentication_mode": "api_key_query_parameter_not_persisted",
        "datacenter_id": DATACENTER,
        "cloud_type": "SECURE",
        "gpu_count": 1,
        "minimum_memory_gb": MIN_MEMORY_GB,
        "preference": list(PREFERENCE),
        "available_candidates": candidates,
        "selected_ordered_gpu_type_ids": ids,
    }
    return ids, evidence


def create_with_capacity_retries(
    create_once,
    *,
    attempts: int = 6,
    delay_seconds: float = 10.0,
):
    observations: list[dict[str, object]] = []
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        ids, evidence = discover_available_gpu_ids()
        observation = {"attempt": attempt, **evidence}
        observations.append(observation)
        if ids:
            try:
                pod = create_once(ids)
                observation["create_status"] = "created"
                observation["created_pod_id"] = pod.get("id") if isinstance(pod, dict) else None
                return pod, observations
            except BaseException as exc:  # capacity can race between query and create
                last_error = exc
                observation["create_status"] = "failed"
                observation["create_error"] = f"{type(exc).__name__}: {exc}"
        else:
            observation["create_status"] = "no_available_capacity"
        if attempt < attempts:
            time.sleep(delay_seconds)
    if last_error is not None:
        raise RuntimeError(f"RunPod capacity-aware Pod creation exhausted retries: {last_error}") from last_error
    raise RuntimeError("RunPod capacity-aware Pod creation found no available Secure Cloud GPU in EU-CZ-1")
