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
                report["issues"].append("current_runpod_balance_endpoint_not_unambiguous")
            else:
                doc = _get(credits[0], key)
                # Fail closed when the API schema cannot be safely interpreted.
                # Never guess whether figures mean cents, USD or available credits.
                report["balance_response_type"] = type(doc).__name__
                report["issues"].append("balance_units_and_available_credit_not_independently_validated")
            try:
                pods = _get("/pods", key)
                items = pods.get("pods") if isinstance(pods, dict) else pods
                report["pod_inventory_count"] = len(items) if isinstance(items, list) else None
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
