#!/usr/bin/env python3
"""Aggregate Phase II role-hardening results without mutating production state."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))

import launch_first_bounded_scientific_training as storage

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/experiments/hephaestus_role_hardening_v2.json"
ROLES = ("controller", "diagnosis", "planner", "evaluator", "judge")


def req(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError("missing required environment variable: " + name)
    return value


def put_json(client: Any, key: str, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    client.put_object(Bucket=storage.VOLUME_ID, Key=key, Body=raw)
    if storage.read_key(client, key) != raw:
        raise RuntimeError("S3 readback mismatch: " + key)


def main() -> int:
    cfg = json.loads(CFG.read_text())
    run_id = req("HEPHAESTUS_ROLE_HARDENING_RUN_ID")
    base = f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    client = storage.s3_client()
    roles: dict[str, Any] = {}
    all_completed = True
    all_accepted = True

    for role in ROLES:
        key = f"{base}/roles/{role}/result.json"
        raw = storage.maybe_read_key(client, key)
        if raw is None:
            roles[role] = {"status": "missing"}
            all_completed = False
            all_accepted = False
            continue
        result = json.loads(raw)
        if result.get("status") != "completed":
            roles[role] = {
                "status": result.get("status"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
            }
            all_completed = False
            all_accepted = False
            continue

        hard = result.get("hardening_certification_summary") or {}
        orig = (result.get("original_certification_regression") or {}).get("observed") or {}
        red = (result.get("frozen_redteam_acceptance") or {}).get("observed") or {}
        accepted = result.get("hardening_accepted") is True
        all_accepted = all_accepted and accepted
        roles[role] = {
            "status": "completed",
            "hardening_accepted": accepted,
            "model_id": result.get("model_id"),
            "revision": result.get("revision"),
            "selected_dev_step": result.get("selected_dev_step"),
            "hardening_cert_quality_100": hard.get("quality_100"),
            "original_cert_quality_100": orig.get("quality_100"),
            "original_cert_baseline_100": cfg["candidates"][role]["original_certification_quality_100"],
            "frozen_redteam_quality_100": red.get("quality_100"),
            "frozen_redteam_baseline_100": cfg["candidates"][role]["original_redteam_quality_100"],
            "frozen_redteam_improvement_points": (
                result.get("frozen_redteam_acceptance") or {}
            ).get("quality_improvement_points"),
            "frozen_redteam_exact_behavior_rate": red.get("exact_behavior_rate"),
            "frozen_redteam_strict_schema_rate": red.get("strict_schema_rate"),
            "frozen_redteam_evidence_precision": red.get("evidence_precision"),
            "frozen_redteam_evidence_recall": red.get("evidence_recall"),
            "frozen_redteam_hallucination_rate": red.get("hallucination_rate"),
            "frozen_redteam_stochastic_triplet_agreement": (
                red.get("stochastic_consistency") or {}
            ).get("triplet_agreement_rate"),
            "selected_adapter": result.get("selected_adapter"),
            "production_promotion_performed": result.get("production_promotion_performed"),
            "production_certification_mutated": result.get("production_certification_mutated"),
            "automatic_role_dispatch_enabled": result.get("automatic_role_dispatch_enabled"),
        }

    aggregate = {
        "result_version": "hephaestus-role-hardening-v2-aggregate.v1",
        "status": "completed" if all_completed else "incomplete",
        "run_id": run_id,
        "pack_sha256": cfg["pack"]["canonical_sha256"],
        "frozen_redteam_pack_sha256": cfg["source_stack"]["original_redteam_pack_sha256"],
        "all_roles_completed": all_completed,
        "all_roles_accepted": bool(all_completed and all_accepted),
        "roles": roles,
        "production_promotion_performed": False,
        "production_certification_mutated": False,
        "automatic_role_dispatch_enabled": False,
        "completed_at_unix": time.time(),
    }
    put_json(client, f"{base}/result.json", aggregate)
    print("ROLE_HARDENING_V2_AGGREGATE_JSON " + json.dumps(aggregate, sort_keys=True))
    return 0 if all_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
