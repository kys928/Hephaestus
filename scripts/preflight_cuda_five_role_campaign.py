#!/usr/bin/env python3
"""RunPod CUDA five-role campaign: read-only preflight with no billing or mutation.

This script deliberately DOES NOT allocate a Pod, infer a balance from historical
UI notifications, issue approvals, or assert that the five role models are
available. Production launch must be separately gated by positive current
credit, region stock, frozen role identities and SHA-verified adapter custody.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.runpod.io/v2"
TARGET = Path(os.environ.get("HEPHAESTUS_CUDA_PREFLIGHT_OUTPUT", "evidence/cuda_preflight.json"))


def _get(path: str, key: str) -> object:
    if not path.startswith("/") or ".." in path:
        raise ValueError("unsafe API path")
    req = urllib.request.Request(
        API + path, headers={"Authorization": "Bearer " + key},
        method="GET")
    with urllib.request.urlopen(req, timeout=25) as res:
        return json.loads(res.read(2_000_000))


def _legacy_readonly_balance(key: str) -> float:
    """Read RunPod's documented current prepaid USD balance, never mutate.

    REST v2 OpenAPI currently has no unambiguous balance endpoint. RunPod's
    authenticated GraphQL myself.clientBalance query is only a temporary
    narrow read-only fallback; it must be removed before GraphQL retirement.
    """
    payload = json.dumps({"query": "query { myself { clientBalance } }"}).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=payload, method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as res:
        obj = json.loads(res.read(50000))
    if not isinstance(obj, dict) or obj.get("errors"):
        raise ValueError("RunPod prepaid-balance query was rejected")
    amount = obj.get("data", {}).get("myself", {}).get("clientBalance")
    if type(amount) not in (float, int) or not math.isfinite(amount):
        raise ValueError("RunPod credit balance is missing or invalid")
    return float(amount)


def _paths(spec: object) -> tuple[list[str], list[str]]:
    if not isinstance(spec, dict) or not isinstance(spec.get("paths"), dict):
        return [], []
    found, credit = [], []
    for path, meta in spec["paths"].items():
        if not isinstance(path, str) or not isinstance(meta, dict):
            continue
        if not isinstance(meta.get("get"), dict) or "{" in path:
            continue
        if any(s in path.casefold() for s in ("billing", "balance", "credit", "account")):
            found.append(path)
            if any(s in path.casefold() for s in ("balance", "credit")):
                credit.append(path)
    return found, credit


def main() -> int:
    report = {
        "schema": "hephaestus-five-role-cuda-preflight.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "read_only_no_pod_created",
        "certification": "not_certified",
        "production_activation": False,
        "head_sha": os.environ.get("GITHUB_SHA", ""),
        "status": "blocked",
        "issues": [],
    }
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        report["issues"].append("github_secret_RUNPOD_API_KEY_unavailable")
    else:
        try:
            # Fetching the official API schema is safe and discovers v2
            # billing endpoints without hard-coding an obsolete GraphQL call.
            spec = _get("/openapi.json", key)
            paths, credits = _paths(spec)
            report["v2_balance_routes"] = credits
            report["v2_billing_routes"] = paths
            report["openapi_sha256"] = "sha256:" + hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if len(credits) != 1:
                report["balance_endpoint"] = "documented_readonly_graphql_fallback"
                try:
                    amount = _legacy_readonly_balance(key)
                    report["current_balance_usd"] = round(amount, 4)
                    report["minimum_new_pod_credit_usd"] = 4.0
                    if amount < 4.0:
                        report["issues"].append("verified_balance_below_new_pod_threshold")
                except (urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
                    report["issues"].append("runpod_current_credit_read_failed:" + type(exc).__name__)
            else:
                report["issues"].append("v2_balance_route_semantics_require_review_before_spending")
            try:
                catalog = _get("/catalog/datacenters/EU-CZ-1?include=GPU_AVAILABILITY", key)
                if isinstance(catalog, dict) and catalog.get("id") == "EU-CZ-1":
                    report["datacenter_gpu_ids_matching_6000"] = [
                        {p: item.get(p) for p in ("id", "availability")}
                        for item in catalog.get("gpuAvailability", [])
                        if isinstance(item, dict) and "6000" in str(item.get("id", ""))
                    ]
                else:
                    report["issues"].append("gpu_catalog_not_scoped_to_pinned_datacenter")
            except (urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
                report["issues"].append("gpu_availability_catalog_read_failed:" + type(exc).__name__)
            try:
                pods = _get("/pods", key)
                items = pods.get("pods") if isinstance(pods, dict) else pods
                report["pod_inventory_count"] = len(items) if isinstance(items, list) else None
                if isinstance(items, list):
                    report["pod_inventory"] = [
                        {key: pod.get(key) for key in ("id", "name", "status", "desiredStatus")
                         if pod.get(key) is not None}
                        for pod in items if isinstance(pod, dict)
                    ][:10]
            except (urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
                report["issues"].append("pod_inventory_read_failed:" + type(exc).__name__)
        except (urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
            report["issues"].append("runpod_v2_read_only_probe_failed:" + type(exc).__name__)
    report["issues"].append("five_frozen_role_adapter_snapshots_not_yet_probed")
    report["issues"].append("cuda_execution_requires_separate_exact_model_admission")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {
        "v2_billing_routes", "openapi_sha256"}}, sort_keys=True))
    return 0  # Safe preflight records a blocked outcome; the CI still passes.


if __name__ == "__main__":
    sys.exit(main())
