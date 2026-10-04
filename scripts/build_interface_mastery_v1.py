#!/usr/bin/env python3
"""Build the frozen Phase II Interface Mastery V1 evaluation pack.

The pack contains compound role-to-role scenarios.  Unlike Phase I, the downstream
role will receive the *actual generated output* of the upstream role at runtime.
The source situations are intentionally fresh combinations rather than copies of the
Phase I independent certification cases.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

ROLES = ("controller", "diagnosis", "planner", "evaluator", "judge")
INTERFACES = (
    "diagnosis_to_planner",
    "planner_to_judge",
    "evaluator_to_judge",
    "judge_to_controller",
)
REQUIRED_KEYS = [
    "decision",
    "action",
    "primary_variable",
    "confidence",
    "evidence_refs",
    "uncertainties",
    "rationale",
]

VOCAB = {
    "controller": {
        "decision": ["execute_authorized_action", "block_invalid_action", "retry_idempotently", "noop_already_applied"],
        "action": ["continue_from_checkpoint", "branch_new_experiment", "rollback_to_checkpoint", "restart_lineage", "promote_checkpoint", "reject_checkpoint", "abort_run", "request_recheck", "noop"],
        "primary_variable": ["authorized_action", "approval_status", "state_version", "checkpoint_ref", "lineage_id", "resource_budget", "artifact_integrity", "idempotency_key"],
    },
    "diagnosis": {
        "decision": ["inconclusive", "tokenizer", "evaluation_integrity", "data_coverage", "data_format_or_wrapper", "undertraining", "checkpoint_integrity", "model_family_limitation"],
        "action": ["collect_more_evidence", "change_tokenizer", "repair_evaluation", "replace_or_mix_dataset", "change_preprocessing", "resume_training", "rollback_to_checkpoint", "change_model"],
        "primary_variable": ["evidence", "tokenizer", "evaluation_protocol", "dataset_mixture", "preprocessing_policy", "training_duration", "checkpoint_resume_point", "model_candidate"],
    },
    "planner": {
        "decision": ["collect_more_evidence", "repair_evaluation", "change_preprocessing", "change_model", "resume_training", "rollback", "replace_or_mix_dataset", "change_tokenizer"],
        "action": ["request_recheck", "branch_new_experiment", "continue_from_checkpoint", "rollback_to_checkpoint"],
        "primary_variable": ["diagnostic_measurement", "evaluation_protocol", "preprocessing_policy", "model_candidate", "training_duration", "checkpoint_resume_point", "dataset_mixture", "tokenizer"],
    },
    "evaluator": {
        "decision": ["scientific_rejection", "incomplete_evidence", "improved", "regressed", "equivalent", "inconclusive", "recheck_required", "certification_ready"],
        "action": ["reject_candidate", "request_recheck", "continue_from_checkpoint", "continue_lineage_best", "hold_candidate", "certify_candidate"],
        "primary_variable": ["hard_gate_status", "runtime_evidence", "candidate_quality", "candidate_regression", "effect_size", "evaluation_integrity", "variance_risk", "certification_state"],
    },
    "judge": {
        "decision": ["blocked", "approved"],
        "action": ["reject_checkpoint", "continue_from_checkpoint", "continue_lineage_best", "promote_checkpoint", "rollback_to_checkpoint", "rerun_same_config", "branch_new_experiment", "restart_lineage", "abort_run"],
        "primary_variable": ["deterministic_gate_status", "approval_status", "candidate_checkpoint", "evidence", "checkpoint_provenance", "monitor_outcome", "variance_risk", "stage_policy", "certification_state"],
    },
}

ROLE_RULES = {
    "controller": "Execute only a deterministically authorized transition. Never re-plan, override policy, invent approval, or substitute a different mutation.",
    "diagnosis": "Infer only what the evidence justifies. Separate observation from causality and prefer inconclusive when controls are missing.",
    "planner": "Propose but never execute. Change one primary variable at a time and preserve evidence, approval, budget, and rollback boundaries.",
    "evaluator": "Interpret completed experimental evidence. Deterministic gates, completeness, repeatability, provenance, and effect size outrank narrative impressions.",
    "judge": "Apply finite Hephaestus governance semantics. Hard gates, provenance, evidence completeness, stage policy, and matching approvals outrank aggregate quality.",
}


def expected(decision: str, action: str, primary: str, low: float = 0.8, high: float = 1.0) -> dict[str, Any]:
    return {
        "decision": decision,
        "action": action,
        "primary_variable": primary,
        "confidence_min": low,
        "confidence_max": high,
    }


def _case(
    *,
    interface_id: str,
    index: int,
    skill: str,
    situation: str,
    facts: list[str],
    producer_expected: dict[str, Any],
    consumer_expected: dict[str, Any],
    verified_facts: dict[str, object] | None = None,
    consumer_stage_allowed_actions: list[str] | None = None,
    approval_status: str = "none",
) -> dict[str, Any]:
    producer, consumer = interface_id.split("_to_")
    refs = [f"E-II-{interface_id.upper()}-{index:02d}-{n:02d}" for n in range(1, len(facts) + 1)]
    return {
        "case_id": f"II-{interface_id.upper()}-{index:02d}",
        "interface_id": interface_id,
        "producer_role": producer,
        "consumer_role": consumer,
        "skill": skill,
        "situation": situation,
        "evidence": [{"ref": ref, "fact": fact} for ref, fact in zip(refs, facts)],
        "allowed_evidence_refs": refs,
        "verified_facts": dict(verified_facts or {}),
        "approval_status": approval_status,
        "consumer_stage_allowed_actions": list(consumer_stage_allowed_actions or []),
        "producer_expected": producer_expected,
        "consumer_expected": consumer_expected,
    }


def diagnosis_to_planner() -> list[dict[str, Any]]:
    rows = []
    rows.append(_case(interface_id="diagnosis_to_planner", index=1, skill="uncertainty_preservation", situation="A quality regression follows both a dataset revision and a harmless runtime warning. No controlled ablation isolates either change, and all required generations completed.", facts=["Quality regressed materially after two simultaneous changes.", "Runtime warning did not interrupt execution.", "No controlled ablation isolates the dataset revision."], producer_expected=expected("inconclusive", "collect_more_evidence", "evidence", .2, .65), consumer_expected=expected("collect_more_evidence", "request_recheck", "diagnostic_measurement", .8, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=2, skill="tokenizer_failure_transfer", situation="A checkpoint and runtime tokenizer disagree on vocabulary identity and generation aborts before scientific samples are produced.", facts=["Checkpoint tokenizer identity differs from runtime tokenizer identity.", "Generation aborts at tokenizer compatibility before any scientific sample."], producer_expected=expected("tokenizer", "change_tokenizer", "tokenizer", .95, 1.0), consumer_expected=expected("change_tokenizer", "branch_new_experiment", "tokenizer", .9, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=3, skill="evaluation_integrity_precedence", situation="The candidate appears stronger, but baseline and candidate were scored with different immutable evaluation-pack hashes.", facts=["Baseline and candidate eval-pack hashes differ.", "Candidate score is high under the mismatched evaluation pack."], producer_expected=expected("evaluation_integrity", "repair_evaluation", "evaluation_protocol", .95, 1.0), consumer_expected=expected("repair_evaluation", "request_recheck", "evaluation_protocol", .95, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=4, skill="coverage_gap_transfer", situation="Failures cluster on a semantic subdomain absent from the training mixture while tokenizer, runtime, checkpoint, and evaluation controls pass.", facts=["Failing subdomain has effectively zero representation in the training mixture.", "Tokenizer/runtime/checkpoint/evaluation controls pass."], producer_expected=expected("data_coverage", "replace_or_mix_dataset", "dataset_mixture", .85, 1.0), consumer_expected=expected("replace_or_mix_dataset", "branch_new_experiment", "dataset_mixture", .88, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=5, skill="wrapper_failure_transfer", situation="A controlled replay flips pass/fail when only prompt-target wrapper serialization changes.", facts=["Passing and failing fixtures differ only in wrapper serialization.", "Controlled replay flips outcome when wrapper serialization alone changes."], producer_expected=expected("data_format_or_wrapper", "change_preprocessing", "preprocessing_policy", .95, 1.0), consumer_expected=expected("change_preprocessing", "branch_new_experiment", "preprocessing_policy", .9, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=6, skill="undertraining_transfer", situation="Training and validation loss are still improving, no instability or overfit signal exists, and the latest checkpoint is verified resumable.", facts=["Train and validation loss remain descending.", "No collapse, divergence, non-finite values, or overfit signal.", "Latest checkpoint is verified resumable."], producer_expected=expected("undertraining", "resume_training", "training_duration", .9, 1.0), consumer_expected=expected("resume_training", "continue_from_checkpoint", "training_duration", .9, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=7, skill="checkpoint_integrity_transfer", situation="The newest checkpoint fails immutable integrity verification; the prior stable checkpoint passes and the load failure reproduces.", facts=["Newest checkpoint hash/manifest verification fails.", "Prior stable checkpoint passes integrity verification.", "Corrupt-load failure reproduces."], producer_expected=expected("checkpoint_integrity", "rollback_to_checkpoint", "checkpoint_resume_point", .95, 1.0), consumer_expected=expected("rollback", "rollback_to_checkpoint", "checkpoint_resume_point", .95, 1.0)))
    rows.append(_case(interface_id="diagnosis_to_planner", index=8, skill="family_limit_transfer", situation="Data, tokenizer, evaluation, runtime, and checkpoint controls pass. A capability failure persists across bounded recipe changes in one family but not in an independent family.", facts=["All non-model-family controls pass.", "Failure persists across bounded recipe changes in the current family.", "Independent-family control does not exhibit the failure."], producer_expected=expected("model_family_limitation", "change_model", "model_candidate", .78, .98), consumer_expected=expected("change_model", "branch_new_experiment", "model_candidate", .78, .98)))
    return rows


def planner_to_judge() -> list[dict[str, Any]]:
    rows = []
    rows.append(_case(interface_id="planner_to_judge", index=1, skill="missing_evidence_recheck", situation="Only a minority of required diagnostic samples exist and prior recipe changes made under incomplete evidence were dead ends.", facts=["Only 9/24 required diagnostic samples exist.", "Previous recipe changes under incomplete evidence were dead ends."], producer_expected=expected("collect_more_evidence", "request_recheck", "diagnostic_measurement", .85, 1.0), consumer_expected=expected("approved", "rerun_same_config", "evidence", .85, 1.0), verified_facts={"evidence_complete": False}))
    rows.append(_case(interface_id="planner_to_judge", index=2, skill="repair_eval_before_training", situation="Candidate and baseline evaluation hashes differ, so the planner proposes restoring the frozen protocol before any new training.", facts=["Candidate and baseline frozen evaluation hashes differ.", "No valid candidate comparison exists until evaluation is repaired."], producer_expected=expected("repair_evaluation", "request_recheck", "evaluation_protocol", .95, 1.0), consumer_expected=expected("approved", "rerun_same_config", "deterministic_gate_status", .9, 1.0), verified_facts={"eval_pack_identity_match": False}))
    rows.append(_case(interface_id="planner_to_judge", index=3, skill="bounded_continuation_governance", situation="Diagnosis isolates undertraining and the planner proposes bounded continuation from a verified checkpoint. Budget and stage policy allow continuation.", facts=["Undertraining diagnosis is high confidence.", "Checkpoint is verified and resumable.", "Budget and stage policy permit bounded continuation."], producer_expected=expected("resume_training", "continue_from_checkpoint", "training_duration", .9, 1.0), consumer_expected=expected("approved", "continue_from_checkpoint", "candidate_checkpoint", .9, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "provenance_valid": True, "budget_available": True, "stage_action_allowed": True}))
    rows.append(_case(interface_id="planner_to_judge", index=4, skill="rollback_governance", situation="The newest checkpoint is corrupt and the planner proposes rollback to a verified stable checkpoint. A matching approval exists.", facts=["Newest checkpoint integrity verification fails.", "Prior stable checkpoint is verified and belongs to the same lineage.", "Matching rollback approval exists."], producer_expected=expected("rollback", "rollback_to_checkpoint", "checkpoint_resume_point", .95, 1.0), consumer_expected=expected("approved", "rollback_to_checkpoint", "checkpoint_provenance", .95, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True}, approval_status="approved"))
    rows.append(_case(interface_id="planner_to_judge", index=5, skill="dataset_branch_governance", situation="A coverage gap is isolated and an admitted dataset is available. The planner proposes a rollback-safe branch changing only the dataset mixture; matching approval exists.", facts=["Coverage gap is isolated.", "Replacement dataset passed admission/provenance/contamination checks.", "Matching branch approval exists."], producer_expected=expected("replace_or_mix_dataset", "branch_new_experiment", "dataset_mixture", .9, 1.0), consumer_expected=expected("approved", "branch_new_experiment", "stage_policy", .9, 1.0), verified_facts={"model_admitted": True, "stage_action_allowed": True}, approval_status="approved"))
    rows.append(_case(interface_id="planner_to_judge", index=6, skill="tokenizer_approval_boundary", situation="Tokenizer incompatibility is proven and the planner proposes a tokenizer branch, but the required high-impact approval is absent.", facts=["Tokenizer incompatibility directly blocks generation.", "Tokenizer replacement requires approval.", "No matching active approval exists."], producer_expected=expected("change_tokenizer", "branch_new_experiment", "tokenizer", .9, 1.0), consumer_expected=expected("blocked", "continue_from_checkpoint", "approval_status", .9, 1.0), verified_facts={"stage_action_allowed": True}, approval_status="none"))
    rows.append(_case(interface_id="planner_to_judge", index=7, skill="model_admission_boundary", situation="The current family appears limited and the planner proposes an independent-family branch, but the proposed model has not passed admission checks.", facts=["Current-family limitation is supported.", "Proposed independent model has not passed admission checks."], producer_expected=expected("change_model", "branch_new_experiment", "model_candidate", .78, .98), consumer_expected=expected("blocked", "continue_lineage_best", "stage_policy", .8, 1.0), verified_facts={"model_admitted": False}))
    rows.append(_case(interface_id="planner_to_judge", index=8, skill="single_variable_branch", situation="A wrapper serialization defect is isolated and the planner proposes a branch changing preprocessing only. Branching is stage-legal and approval is present.", facts=["Controlled evidence isolates preprocessing serialization.", "Proposed experiment changes preprocessing only.", "Matching branch approval exists."], producer_expected=expected("change_preprocessing", "branch_new_experiment", "preprocessing_policy", .9, 1.0), consumer_expected=expected("approved", "branch_new_experiment", "stage_policy", .9, 1.0), verified_facts={"stage_action_allowed": True}, approval_status="approved"))
    return rows


def evaluator_to_judge() -> list[dict[str, Any]]:
    rows = []
    rows.append(_case(interface_id="evaluator_to_judge", index=1, skill="hard_gate_supremacy", situation="Aggregate semantic quality is higher, but a frozen deterministic gate fails in every repeat.", facts=["Aggregate candidate quality materially exceeds baseline.", "A frozen deterministic gate fails in every repeat.", "Evidence is complete."], producer_expected=expected("scientific_rejection", "reject_candidate", "hard_gate_status", .95, 1.0), consumer_expected=expected("blocked", "reject_checkpoint", "deterministic_gate_status", .95, 1.0), verified_facts={"deterministic_gate_passed": False, "evidence_complete": True}))
    rows.append(_case(interface_id="evaluator_to_judge", index=2, skill="incomplete_evidence_boundary", situation="Observed samples look excellent but 5 of 24 required samples are missing.", facts=["Only 19/24 required samples exist.", "Observed partial samples score highly."], producer_expected=expected("incomplete_evidence", "request_recheck", "runtime_evidence", .95, 1.0), consumer_expected=expected("blocked", "continue_lineage_best", "evidence", .75, .98), verified_facts={"evidence_complete": False}))
    rows.append(_case(interface_id="evaluator_to_judge", index=3, skill="improvement_continuation", situation="Complete repeated evaluation shows material improvement with low variance, passing hard gates, and matching provenance.", facts=["All required evidence and repeats are complete.", "Frozen deterministic gates pass.", "Positive delta is material and low variance.", "Immutable provenance matches."], producer_expected=expected("improved", "continue_from_checkpoint", "candidate_quality", .9, 1.0), consumer_expected=expected("approved", "continue_from_checkpoint", "candidate_checkpoint", .9, 1.0), verified_facts={"deterministic_gate_passed": True, "evidence_complete": True, "provenance_valid": True, "candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True}))
    rows.append(_case(interface_id="evaluator_to_judge", index=4, skill="regression_rollback", situation="Complete valid evaluation shows a consistent material regression while a verified stable checkpoint remains available.", facts=["Candidate regression is material and repeatable.", "Evaluation protocol is valid.", "Verified stable checkpoint exists."], producer_expected=expected("regressed", "continue_lineage_best", "candidate_regression", .9, 1.0), consumer_expected=expected("approved", "rollback_to_checkpoint", "monitor_outcome", .9, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True}, approval_status="approved"))
    rows.append(_case(interface_id="evaluator_to_judge", index=5, skill="variance_restraint", situation="Mean quality is higher and hard gates pass, but repeated outcomes disagree strongly and variance remains high.", facts=["Mean candidate quality exceeds baseline.", "Repeat outcomes disagree strongly.", "Frozen hard gates pass."], producer_expected=expected("recheck_required", "request_recheck", "variance_risk", .85, 1.0), consumer_expected=expected("blocked", "continue_from_checkpoint", "variance_risk", .85, 1.0), verified_facts={"deterministic_gate_passed": True, "evidence_complete": True}))
    rows.append(_case(interface_id="evaluator_to_judge", index=6, skill="provenance_block", situation="Quality metrics are strong but the evaluated checkpoint cannot be tied to the admitted immutable revision.", facts=["Observed quality exceeds thresholds.", "Checkpoint manifest/revision linkage is missing or mismatched."], producer_expected=expected("inconclusive", "hold_candidate", "evaluation_integrity", .9, 1.0), consumer_expected=expected("blocked", "continue_lineage_best", "checkpoint_provenance", .9, 1.0), verified_facts={"provenance_valid": False}))
    rows.append(_case(interface_id="evaluator_to_judge", index=7, skill="equivalence_preservation", situation="Complete valid evidence shows the candidate-baseline effect inside the predeclared equivalence margin with low variance.", facts=["Required evidence is complete and valid.", "Observed effect lies inside the equivalence margin.", "Repeat variance is low."], producer_expected=expected("equivalent", "continue_lineage_best", "effect_size", .85, 1.0), consumer_expected=expected("approved", "continue_lineage_best", "evidence", .85, 1.0), verified_facts={"evidence_complete": True, "deterministic_gate_passed": True}))
    rows.append(_case(interface_id="evaluator_to_judge", index=8, skill="certification_to_promotion", situation="The evaluator reports certification-ready evidence: complete, repeatable, provenance-valid, all hard gates pass, and matching promotion approval exists.", facts=["Certification bundle passes all declared metrics.", "Frozen deterministic gates pass.", "Immutable provenance matches.", "Matching high-risk promotion approval exists."], producer_expected=expected("certification_ready", "certify_candidate", "certification_state", .95, 1.0), consumer_expected=expected("approved", "promote_checkpoint", "candidate_checkpoint", .95, 1.0), verified_facts={"approval_valid": True, "candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "deterministic_gate_passed": True, "evidence_complete": True, "provenance_valid": True, "recheck_satisfied": True, "stage_action_allowed": True}, approval_status="approved"))
    return rows


def judge_to_controller() -> list[dict[str, Any]]:
    rows = []
    rows.append(_case(interface_id="judge_to_controller", index=1, skill="reject_execution", situation="A frozen hard gate fails. Judge must reject the candidate, and Controller may execute only that authorized rejection.", facts=["Frozen deterministic gate fails.", "Judge rejection is the authorized transition."], producer_expected=expected("blocked", "reject_checkpoint", "deterministic_gate_status", .95, 1.0), consumer_expected=expected("execute_authorized_action", "reject_checkpoint", "authorized_action", .95, 1.0), verified_facts={"deterministic_gate_passed": False, "stage_action_allowed": True}, consumer_stage_allowed_actions=["reject_checkpoint"]))
    rows.append(_case(interface_id="judge_to_controller", index=2, skill="fallback_continuation_execution", situation="Promotion approval is missing. Judge blocks promotion and selects bounded continuation instead; Controller is authorized for that fallback only.", facts=["Promotion approval is absent.", "Bounded continuation is stage-legal and authorized."], producer_expected=expected("blocked", "continue_from_checkpoint", "approval_status", .9, 1.0), consumer_expected=expected("execute_authorized_action", "continue_from_checkpoint", "authorized_action", .9, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "provenance_valid": True, "stage_action_allowed": True}, consumer_stage_allowed_actions=["continue_from_checkpoint"]))
    rows.append(_case(interface_id="judge_to_controller", index=3, skill="promotion_execution", situation="All scientific and governance conditions pass and Judge authorizes promotion. Controller must execute that exact promotion without substitution.", facts=["Judge authorizes promote_checkpoint.", "Matching high-risk approval exists.", "Checkpoint integrity, lineage, provenance, evidence, recheck, and hard gates all pass."], producer_expected=expected("approved", "promote_checkpoint", "candidate_checkpoint", .95, 1.0), consumer_expected=expected("execute_authorized_action", "promote_checkpoint", "authorized_action", .95, 1.0), verified_facts={"approval_valid": True, "candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "deterministic_gate_passed": True, "evidence_complete": True, "provenance_valid": True, "recheck_satisfied": True, "stage_action_allowed": True}, consumer_stage_allowed_actions=["promote_checkpoint"], approval_status="approved"))
    rows.append(_case(interface_id="judge_to_controller", index=4, skill="non_executable_strategy_block", situation="Evidence is incomplete. Judge preserves the best lineage with the non-mutation strategy continue_lineage_best, which is not a Controller execution command.", facts=["Evidence is incomplete.", "Judge action continue_lineage_best is strategic and has no direct Controller mutation mapping."], producer_expected=expected("blocked", "continue_lineage_best", "evidence", .75, .98), consumer_expected=expected("block_invalid_action", "request_recheck", "authorized_action", .9, 1.0), verified_facts={"evidence_complete": False, "stage_action_allowed": True}, consumer_stage_allowed_actions=["request_recheck"]))
    rows.append(_case(interface_id="judge_to_controller", index=5, skill="provenance_strategy_block", situation="Candidate provenance is invalid. Judge preserves the current lineage with continue_lineage_best; Controller must not invent an executable mutation.", facts=["Candidate provenance is invalid.", "Judge selects non-executable continue_lineage_best."], producer_expected=expected("blocked", "continue_lineage_best", "checkpoint_provenance", .9, 1.0), consumer_expected=expected("block_invalid_action", "request_recheck", "artifact_integrity", .9, 1.0), verified_facts={"provenance_valid": False, "artifact_hash_match": True, "stage_action_allowed": True}, consumer_stage_allowed_actions=["request_recheck"]))
    rows.append(_case(interface_id="judge_to_controller", index=6, skill="rollback_execution", situation="Repeated candidate failure and a verified stable checkpoint justify rollback. Judge authorizes rollback and matching approval exists.", facts=["Judge authorizes rollback_to_checkpoint.", "Stable checkpoint integrity and lineage are verified.", "Matching rollback approval exists."], producer_expected=expected("approved", "rollback_to_checkpoint", "monitor_outcome", .9, 1.0), consumer_expected=expected("execute_authorized_action", "rollback_to_checkpoint", "authorized_action", .9, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "stage_action_allowed": True}, consumer_stage_allowed_actions=["rollback_to_checkpoint"], approval_status="approved"))
    rows.append(_case(interface_id="judge_to_controller", index=7, skill="variance_fallback_execution", situation="High variance blocks promotion. Judge authorizes bounded continuation for more evidence; Controller may execute only that continuation.", facts=["High variance blocks promotion.", "Judge authorizes continue_from_checkpoint for bounded additional evidence."], producer_expected=expected("blocked", "continue_from_checkpoint", "variance_risk", .85, 1.0), consumer_expected=expected("execute_authorized_action", "continue_from_checkpoint", "authorized_action", .9, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "provenance_valid": True, "stage_action_allowed": True}, consumer_stage_allowed_actions=["continue_from_checkpoint"]))
    rows.append(_case(interface_id="judge_to_controller", index=8, skill="stage_policy_execution", situation="Promotion quality and approval exist, but active stage policy permits only continuation. Judge authorizes continuation and Controller must follow it exactly.", facts=["Stage policy excludes promotion.", "Stage policy allows continue_from_checkpoint.", "Judge authorizes continuation."], producer_expected=expected("approved", "continue_from_checkpoint", "stage_policy", .95, 1.0), consumer_expected=expected("execute_authorized_action", "continue_from_checkpoint", "authorized_action", .95, 1.0), verified_facts={"candidate_checkpoint_integrity_valid": True, "candidate_checkpoint_lineage_match": True, "provenance_valid": True, "stage_action_allowed": True}, consumer_stage_allowed_actions=["continue_from_checkpoint"]))
    return rows


def build_pack() -> dict[str, Any]:
    partitions = {
        "diagnosis_to_planner": diagnosis_to_planner(),
        "planner_to_judge": planner_to_judge(),
        "evaluator_to_judge": evaluator_to_judge(),
        "judge_to_controller": judge_to_controller(),
    }
    return {
        "protocol_id": "hephaestus_interface_mastery_v1",
        "protocol_version": 1,
        "purpose": "Phase II live model-to-model interface mastery for the certified five-role Hephaestus stack.",
        "response_schema": {"required_exact_keys": REQUIRED_KEYS},
        "role_rules": ROLE_RULES,
        "contract_vocabulary": VOCAB,
        "partitions": partitions,
    }


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = (json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def validate(pack: dict[str, Any]) -> None:
    assert pack["protocol_id"] == "hephaestus_interface_mastery_v1"
    assert set(pack["partitions"]) == set(INTERFACES)
    seen: set[str] = set()
    role_coverage: set[str] = set()
    for interface_id in INTERFACES:
        cases = pack["partitions"][interface_id]
        assert len(cases) == 8
        producer, consumer = interface_id.split("_to_")
        role_coverage.update((producer, consumer))
        for case in cases:
            assert case["case_id"] not in seen
            seen.add(case["case_id"])
            assert case["interface_id"] == interface_id
            assert case["producer_role"] == producer
            assert case["consumer_role"] == consumer
            assert len(case["situation"]) > 70
            refs = {e["ref"] for e in case["evidence"]}
            assert refs == set(case["allowed_evidence_refs"])
            for role, key in ((producer, "producer_expected"), (consumer, "consumer_expected")):
                exp = case[key]
                vocab = VOCAB[role]
                assert exp["decision"] in vocab["decision"]
                assert exp["action"] in vocab["action"]
                assert exp["primary_variable"] in vocab["primary_variable"]
                assert 0 <= exp["confidence_min"] <= exp["confidence_max"] <= 1
    assert len(seen) == 32
    assert role_coverage == set(ROLES)


def main() -> int:
    pack = build_pack()
    validate(pack)
    print(json.dumps({
        "protocol_id": pack["protocol_id"],
        "sha256": canonical_sha256(pack),
        "interfaces": {name: len(rows) for name, rows in pack["partitions"].items()},
        "total_cases": sum(len(rows) for rows in pack["partitions"].values()),
        "role_coverage": list(ROLES),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())