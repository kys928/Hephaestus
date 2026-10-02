#!/usr/bin/env python3
"""Select an Evaluator semantic-invariance repair checkpoint from fixed dev evidence."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import run_interface_repair_v1 as v1

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_evaluator_semantic_invariance_repair_v1.json"


def load_cfg() -> dict[str, Any]:
    value = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("repair config must be a JSON object")
    return value


def get_json(client: Any, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        value = json.loads(response["Body"].read().decode("utf-8"))
    finally:
        response["Body"].close()
    if not isinstance(value, dict):
        raise RuntimeError(f"S3 object is not JSON object: {key}")
    return value


def score_key(result: Mapping[str, Any]) -> tuple[float, ...]:
    summary = result["development_summary"]
    analysis = result["development_analysis"]
    per_state = [float(v) for v in analysis["per_state_decision_accuracy"].values()]
    return (
        float(summary["semantic_exact_pass_rate"]),
        float(summary["decision_exact_pass_rate"]),
        min(per_state),
        float(summary["semantic_quality_100"]),
        float(summary["schema_compliance"]),
        float(summary["evidence_grounding"]),
        -float(summary["hallucination_rate"]),
        -float(summary["upstream_copy_violation_rate"]),
        -float(result["checkpoint_step"]),
    )


def main() -> int:
    cfg = load_cfg()
    run_id = (os.environ.get("HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID") or "").strip()
    repo_sha = (os.environ.get("GITHUB_SHA") or "").strip()
    if not run_id or not repo_sha:
        raise RuntimeError("HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID and GITHUB_SHA are required")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())

    baseline = get_json(client, str(cfg["baseline"]["s3_key"]))
    if baseline.get("status") != "baseline_complete":
        raise RuntimeError("recorded frozen baseline is not complete")
    if baseline.get("development_diagnostic_sha256") != cfg["pack"]["development_diagnostic_sha256"]:
        raise RuntimeError("baseline development diagnostic SHA mismatch")
    if baseline.get("development_case_order_sha256") != cfg["pack"]["development_case_order_sha256"]:
        raise RuntimeError("baseline development case order SHA mismatch")

    training = get_json(client, f"{prefix}/training/result.json")
    if training.get("status") != "training_complete" or training.get("repo_sha") != repo_sha:
        raise RuntimeError("training result is missing, incomplete, or bound to another repository SHA")
    if training.get("parent_adapter_sha256") != cfg["required_parent_adapter_sha256"]:
        raise RuntimeError("training parent adapter digest mismatch")
    if training.get("burned_sealed_partition_used") is not False or training.get("development_diagnostic_touched") is not False:
        raise RuntimeError("training cleanliness invariant violated")

    evaluations: list[dict[str, Any]] = []
    for step in [int(x) for x in cfg["training"]["checkpoint_steps"]]:
        key = f"{prefix}/checkpoints/step-{step:03d}/evaluation/result.json"
        result = get_json(client, key)
        if result.get("status") != "evaluation_complete":
            raise RuntimeError(f"checkpoint {step} evaluation is not complete")
        if result.get("repo_sha") != repo_sha or int(result.get("checkpoint_step", -1)) != step:
            raise RuntimeError(f"checkpoint {step} identity mismatch")
        if result.get("development_pack_sha256") != cfg["pack"]["canonical_sha256"]:
            raise RuntimeError(f"checkpoint {step} curriculum SHA mismatch")
        if result.get("development_diagnostic_sha256") != cfg["pack"]["development_diagnostic_sha256"]:
            raise RuntimeError(f"checkpoint {step} diagnostic SHA mismatch")
        if result.get("development_case_order_sha256") != cfg["pack"]["development_case_order_sha256"]:
            raise RuntimeError(f"checkpoint {step} case-order SHA mismatch")
        if result.get("burned_sealed_partition_used") is not False:
            raise RuntimeError(f"checkpoint {step} touched burned sealed evidence")
        evaluations.append(result)

    passing = [r for r in evaluations if bool(r["checkpoint_selection_gates"]["passed"])]
    best_observed = max(evaluations, key=score_key)
    selected = min(passing, key=lambda r: int(r["checkpoint_step"])) if passing else None
    checkpoint_table = []
    for result in sorted(evaluations, key=lambda r: int(r["checkpoint_step"])):
        summary = result["development_summary"]
        checkpoint_table.append({
            "step": int(result["checkpoint_step"]),
            "gates_passed": bool(result["checkpoint_selection_gates"]["passed"]),
            "decision_exact_pass_rate": float(summary["decision_exact_pass_rate"]),
            "semantic_exact_pass_rate": float(summary["semantic_exact_pass_rate"]),
            "semantic_quality_100": float(summary["semantic_quality_100"]),
            "schema_compliance": float(summary["schema_compliance"]),
            "evidence_grounding": float(summary["evidence_grounding"]),
            "hallucination_rate": float(summary["hallucination_rate"]),
            "system_action_exact_pass_rate": float(summary["system_action_exact_pass_rate"]),
            "upstream_copy_violation_rate": float(summary["upstream_copy_violation_rate"]),
            "per_state_decision_accuracy": result["development_analysis"]["per_state_decision_accuracy"],
            "adapter": result["checkpoint_adapter"],
        })

    selection: dict[str, Any] = {
        "result_version": "hephaestus-evaluator-semantic-invariance-repair-selection.v1",
        "run_id": run_id,
        "repo_sha": repo_sha,
        "parent_candidate_id": cfg["required_parent_candidate_id"],
        "parent_adapter_sha256": cfg["required_parent_adapter_sha256"],
        "development_pack_sha256": cfg["pack"]["canonical_sha256"],
        "development_diagnostic_sha256": cfg["pack"]["development_diagnostic_sha256"],
        "development_case_order_sha256": cfg["pack"]["development_case_order_sha256"],
        "baseline": {
            "run_id": cfg["baseline"]["run_id"],
            "decision_exact_pass_rate": float(baseline["development_summary"]["decision_exact_pass_rate"]),
            "semantic_exact_pass_rate": float(baseline["development_summary"]["semantic_exact_pass_rate"]),
        },
        "checkpoint_results": checkpoint_table,
        "all_checkpoint_evaluations_complete": True,
        "burned_sealed_partition_used": False,
        "fresh_certification_generated": False,
        "certification_claim_performed": False,
        "production_promotion_performed": False,
        "automatic_role_dispatch_performed": False,
        "selected_at_unix": time.time(),
    }
    if selected is None:
        selection.update({
            "status": "scientifically_rejected_no_checkpoint_cleared_gates",
            "selected_checkpoint": None,
            "best_observed_nonselectable": {
                "step": int(best_observed["checkpoint_step"]),
                "adapter": best_observed["checkpoint_adapter"],
                "development_summary": best_observed["development_summary"],
                "development_analysis": best_observed["development_analysis"],
                "failed_checks": [
                    key for key, passed in best_observed["checkpoint_selection_gates"]["checks"].items() if not passed
                ],
            },
        })
    else:
        selection.update({
            "status": "checkpoint_selected",
            "selected_checkpoint": {
                "step": int(selected["checkpoint_step"]),
                "adapter": selected["checkpoint_adapter"],
                "development_summary": selected["development_summary"],
                "development_analysis": selected["development_analysis"],
                "checkpoint_selection_gates": selected["checkpoint_selection_gates"],
            },
            "best_observed_nonselectable": None,
        })
    v1.put_json(client, f"{prefix}/selection.json", selection)
    print("EVALUATOR_SEMANTIC_REPAIR_SELECTION_JSON " + json.dumps(selection, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
