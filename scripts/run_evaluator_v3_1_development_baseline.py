#!/usr/bin/env python3
"""Evaluate frozen Evaluator V3.1 once on the semantic-invariance development diagnostic.

This is an inference-only development baseline. It does not train, mutate or persist
weights, certify a candidate, promote production state, or read the burned sealed
certification partition. The exact 256-case development diagnostic is hashed and
persisted so later repair checkpoints can be compared against the same reference.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import build_evaluator_semantic_invariance_repair_v1 as dev
import run_evaluator_sealed_certification_v1 as sealed
import run_interface_repair_v1 as v1
from interface_repair_v2_bootstrap import materialize_parent_cached

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.json"
ROLE = "evaluator"
EXPECTED_PACK_SHA256 = "0d01fe1853742d1259a333fa9a92dec46e8789b69f2019ed34e17d815fc01a3c"
EXPECTED_CANDIDATE_ID = "evaluator-v3.1-boundary-v1"
EXPECTED_BASE_REVISION = "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
EXPECTED_ADAPTER_SHA256 = "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"
S3_ROOT = "hephaestus/scientific/v4/evaluator_development_baseline"
WEAK_STATES = {"improved", "equivalent", "inconclusive", "recheck_required"}
ROBUST_STATES = {"certification_ready", "incomplete_evidence", "regressed", "scientific_rejection"}


def load_cfg() -> dict[str, Any]:
    cfg = sealed.load_json(CFG_PATH)
    governance = cfg["governance"]
    if governance.get("training_allowed") is not False:
        raise RuntimeError("development baseline may not train")
    if governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("development baseline may not mutate adapters")
    if governance.get("candidate_weights_mutation_allowed") is not False:
        raise RuntimeError("development baseline may not mutate candidate weights")
    if governance.get("production_promotion_allowed") or governance.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("development baseline may not promote or enable dispatch")
    return cfg


def load_candidate(cfg: Mapping[str, Any]) -> dict[str, Any]:
    candidate = sealed.load_json(ROOT / str(cfg["candidate_manifest"]))
    if candidate.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise RuntimeError("candidate identity drift")
    model = candidate.get("model", {})
    governance = candidate.get("governance", {})
    if governance.get("weights_frozen") is not True or governance.get("training_allowed") is not False:
        raise RuntimeError("candidate weights are not frozen")
    if governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("candidate adapter mutation is not prohibited")
    if model.get("revision") != EXPECTED_BASE_REVISION:
        raise RuntimeError("base revision drift")
    if model.get("adapter_sha256") != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("V3.1 adapter digest drift")
    return candidate


def load_development_pack() -> tuple[dict[str, Any], str, str]:
    pack = dev.build_pack()
    dev.validate(pack)
    pack_sha = dev.canonical_sha256(pack)
    if pack_sha != EXPECTED_PACK_SHA256:
        raise RuntimeError(f"development curriculum hash drift: {pack_sha} != {EXPECTED_PACK_SHA256}")
    diagnostic = list(pack["partitions"][ROLE]["development_diagnostic"])
    if len(diagnostic) != 256:
        raise RuntimeError(f"development diagnostic must contain 256 cases, got {len(diagnostic)}")
    counts = Counter(str(case["semantic_root"]) for case in diagnostic)
    expected_counts = Counter({state: 32 for state in dev.STATE_ORDER})
    if counts != expected_counts:
        raise RuntimeError(f"development state distribution drift: {dict(counts)}")
    raw = json.dumps(diagnostic, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    diagnostic_sha = hashlib.sha256(raw).hexdigest()
    ordered_ids = [str(case["case_id"]) for case in diagnostic]
    order_sha = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    return pack, diagnostic_sha, order_sha


def development_gates(summary: Mapping[str, float], analysis: Mapping[str, Any]) -> dict[str, Any]:
    per_state = dict(analysis["per_state_decision_accuracy"])
    checks: dict[str, bool] = {
        "overall_decision_exact_gte_95pct": float(summary["decision_exact_pass_rate"]) >= 0.95,
        "overall_semantic_exact_gte_95pct": float(summary["semantic_exact_pass_rate"]) >= 0.95,
        "schema_100pct": float(summary["schema_compliance"]) >= 1.0,
        "grounding_gte_99pct": float(summary["evidence_grounding"]) >= 0.99,
        "hallucination_lte_1pct": float(summary["hallucination_rate"]) <= 0.01,
        "primary_projection_100pct": float(summary["primary_variable_projection_rate"]) >= 1.0,
        "boundary_projection_100pct": float(summary["boundary_projection_rate"]) >= 1.0,
        "system_action_exact_100pct": float(summary["system_action_exact_pass_rate"]) >= 1.0,
        "system_escalation_zero": float(summary["system_semantic_escalation_rate"]) <= 0.0,
        "upstream_copy_lte_1pct": float(summary["upstream_copy_violation_rate"]) <= 0.01,
    }
    for state in sorted(WEAK_STATES):
        checks[f"weak_state_{state}_gte_90pct"] = float(per_state.get(state, 0.0)) >= 0.90
    for state in sorted(ROBUST_STATES):
        checks[f"robust_state_{state}_gte_95pct"] = float(per_state.get(state, 0.0)) >= 0.95
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    cfg = load_cfg()
    candidate = load_candidate(cfg)
    run_id = (os.environ.get("HEPHAESTUS_EVALUATOR_DEV_BASELINE_RUN_ID") or "").strip()
    if not run_id:
        raise RuntimeError("HEPHAESTUS_EVALUATOR_DEV_BASELINE_RUN_ID is required")
    prefix = f"{S3_ROOT}/{run_id}"
    progress_key = f"{prefix}/progress.json"
    result_key = f"{prefix}/result.json"
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())
    deadline = time.monotonic() + float(cfg["execution"]["hard_wall_seconds"])
    started_at = time.time()
    development_touched = False

    def heartbeat(stage: str, **extra: Any) -> None:
        payload = {
            "run_id": run_id,
            "stage": stage,
            "timestamp_unix": time.time(),
            "remaining_wall_seconds": max(0.0, deadline - time.monotonic()),
            **extra,
        }
        v1.put_json(client, progress_key, payload)
        print("EVALUATOR_DEV_BASELINE_PROGRESS_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        heartbeat("candidate_verification_started")
        observed_blobs = sealed.verify_code_blobs(candidate)
        pack, diagnostic_sha, order_sha = load_development_pack()
        role_spec = sealed.frozen_role_spec(cfg, candidate)
        root = Path("/opt/hephaestus-evaluator-v3-1-development-baseline")
        root.mkdir(parents=True, exist_ok=True)
        heartbeat(
            "materializing_frozen_candidate",
            adapter_sha256=role_spec["adapter"]["sha256"],
            development_pack_sha256=EXPECTED_PACK_SHA256,
            development_diagnostic_sha256=diagnostic_sha,
            development_case_order_sha256=order_sha,
        )
        base_dir, adapter_dir = materialize_parent_cached(client, role_spec, root)
        model, tokenizer, runtime_info = v1.load_model(
            ROLE, sealed._model_cfg(cfg), role_spec, base_dir, adapter_dir
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 0:
            raise RuntimeError("development baseline unexpectedly exposes trainable parameters")

        preflight_rows, preflight_summary = sealed.evaluate_partition(
            cfg=cfg,
            pack=pack,
            partition="preflight",
            model=model,
            tokenizer=tokenizer,
            deadline=deadline,
            client=client,
            prefix=prefix,
            heartbeat=heartbeat,
        )
        preflight_passed = (
            preflight_summary["final_complete_contract_json_rate"] >= float(cfg["preflight"]["require_complete_contract_json_rate"])
            and preflight_summary["schema_compliance"] >= float(cfg["preflight"]["require_schema_compliance"])
        )
        preflight_record = {
            "passed": preflight_passed,
            "sample_count": len(preflight_rows),
            "final_complete_contract_json_rate": preflight_summary["final_complete_contract_json_rate"],
            "initial_complete_contract_json_rate": preflight_summary["initial_complete_contract_json_rate"],
            "schema_compliance": preflight_summary["schema_compliance"],
            "format_retry_attempt_rate": preflight_summary["format_retry_attempt_rate"],
            "format_retry_success_rate": preflight_summary["format_retry_success_rate"],
        }
        if not preflight_passed:
            result = {
                "result_version": "hephaestus-evaluator-v3.1-development-baseline.v1",
                "status": "preflight_rejected",
                "run_id": run_id,
                "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
                "candidate_id": candidate["candidate_id"],
                "candidate_adapter_sha256": candidate["model"]["adapter_sha256"],
                "development_pack_sha256": EXPECTED_PACK_SHA256,
                "development_diagnostic_sha256": diagnostic_sha,
                "development_case_order_sha256": order_sha,
                "development_diagnostic_touched": False,
                "training_performed": False,
                "adapter_mutated": False,
                "certification_claim_performed": False,
                "production_promotion_performed": False,
                "preflight": preflight_record,
                "runtime": {**runtime_info, "trainable_parameter_count_after_freeze": 0},
                "completed_at_unix": time.time(),
            }
            v1.put_json(client, result_key, result)
            print("EVALUATOR_DEV_BASELINE_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
            return 3

        development_touched = True
        heartbeat("development_diagnostic_started", sample_count=256)
        rows, summary = sealed.evaluate_partition(
            cfg=cfg,
            pack=pack,
            partition="development_diagnostic",
            model=model,
            tokenizer=tokenizer,
            deadline=deadline,
            client=client,
            prefix=prefix,
            heartbeat=heartbeat,
        )
        analysis = sealed.decision_analysis(rows)
        gates = development_gates(summary, analysis)
        state_counts = dict(sorted(Counter(str(row["semantic_root"]) for row in rows).items()))
        result = {
            "result_version": "hephaestus-evaluator-v3.1-development-baseline.v1",
            "status": "baseline_complete",
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "candidate_id": candidate["candidate_id"],
            "weights_lineage": candidate.get("weights_lineage"),
            "candidate_adapter_sha256": candidate["model"]["adapter_sha256"],
            "base_model_id": candidate["model"]["model_id"],
            "base_model_revision": candidate["model"]["revision"],
            "development_pack_sha256": EXPECTED_PACK_SHA256,
            "development_diagnostic_sha256": diagnostic_sha,
            "development_case_order_sha256": order_sha,
            "development_case_count": len(rows),
            "development_state_counts": state_counts,
            "training_performed": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "adapter_mutated": False,
            "adapter_persisted": False,
            "certification_claim_performed": False,
            "production_promotion_performed": False,
            "automatic_role_dispatch_performed": False,
            "burned_sealed_partition_used": False,
            "preflight": preflight_record,
            "development_summary": summary,
            "development_analysis": analysis,
            "checkpoint_selection_gates": gates,
            "runtime": {**runtime_info, "trainable_parameter_count_after_freeze": 0},
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
        }
        v1.put_json(client, result_key, result)
        heartbeat(
            "complete",
            decision_exact=summary["decision_exact_pass_rate"],
            semantic_exact=summary["semantic_exact_pass_rate"],
            development_gate_passed=gates["passed"],
        )
        print("EVALUATOR_DEV_BASELINE_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failed = {
            "result_version": "hephaestus-evaluator-v3.1-development-baseline.v1",
            "status": "failed",
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "candidate_id": candidate.get("candidate_id"),
            "development_pack_sha256": EXPECTED_PACK_SHA256,
            "development_diagnostic_touched": development_touched,
            "training_performed": False,
            "adapter_mutated": False,
            "certification_claim_performed": False,
            "production_promotion_performed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_at_unix": time.time(),
        }
        try:
            v1.put_json(client, result_key, failed)
        except Exception:
            pass
        print("EVALUATOR_DEV_BASELINE_RESULT_JSON " + json.dumps(failed, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
