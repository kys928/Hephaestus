#!/usr/bin/env python3
"""Build targeted Phase II repair data without touching either certification holdout.

The repair corpus is derived from observed *failure classes*, not from Phase II case
text.  Every semantic sample stores both raw and normalized handoff representations
so the normalization A/B result can choose the eventual training presentation without
regenerating labels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from hephaestus.evaluation.handoff_normalization import normalize_handoff

ROOT = Path(__file__).resolve().parents[1]
V1_BUILDER = ROOT / "scripts/build_interface_mastery_v1.py"
ROLES = ("controller", "evaluator", "judge")
SAMPLES_PER_ROLE = 40


def _v1():
    spec = importlib.util.spec_from_file_location("interface_mastery_v1_contract", V1_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase II V1 contract vocabulary")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected(decision: str, action: str, primary: str, confidence: float, refs: list[str]) -> dict[str, Any]:
    return {
        "decision": decision,
        "action": action,
        "primary_variable": primary,
        "confidence": confidence,
        "evidence_refs": refs,
        "uncertainties": [],
        "rationale": "The target-role contract and verified facts determine this bounded output.",
    }


def _sample(
    *,
    target_role: str,
    source_role: str,
    family: str,
    index: int,
    situation: str,
    facts: list[str],
    upstream: dict[str, Any],
    target: dict[str, Any],
    verified_facts: dict[str, object],
) -> dict[str, Any]:
    sid = f"IR-{target_role.upper()}-{family.upper().replace('_','-')}-{index:02d}"
    refs = [f"E-{sid}-{n:02d}" for n in range(1, len(facts) + 1)]
    upstream = dict(upstream)
    upstream["evidence_refs"] = refs
    upstream.setdefault("uncertainties", [])
    upstream.setdefault("rationale", "Upstream advisory record.")
    target = dict(target)
    target["evidence_refs"] = refs
    raw = json.dumps(upstream, sort_keys=True, separators=(",", ":"))
    return {
        "sample_id": sid,
        "target_role": target_role,
        "split": "train" if index <= 8 else "dev",
        "source_role": source_role,
        "failure_family": family,
        "situation": situation,
        "evidence": [{"ref": ref, "fact": fact} for ref, fact in zip(refs, facts)],
        "allowed_evidence_refs": refs,
        "verified_facts": verified_facts,
        "raw_handoff": raw,
        "normalized_handoff": normalize_handoff(source_role=source_role, raw_output=raw),
        "target_output": target,
        "training_use": "targeted_interface_repair_only",
    }


def controller_samples(vocab: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legal_actions = [
        ("continue_from_checkpoint", "candidate_checkpoint"),
        ("rollback_to_checkpoint", "checkpoint_provenance"),
        ("branch_new_experiment", "stage_policy"),
        ("promote_checkpoint", "certification_state"),
    ]
    for i in range(1, 11):
        action, upstream_primary = legal_actions[(i - 1) % len(legal_actions)]
        rows.append(_sample(
            target_role="controller", source_role="judge", family="translate_not_copy", index=i,
            situation=f"Judge authorizes {action} in repair scenario {i}; the Controller must translate the advisory record into its own execution contract rather than copy Judge's primary variable.",
            facts=["The requested action is stage-legal.", "Checkpoint/provenance checks required for the action pass.", "The Judge record is advisory, not executable state."],
            upstream={"decision": "approved", "action": action, "primary_variable": upstream_primary, "confidence": 0.96},
            target=_expected("execute_authorized_action", action, "authorized_action", 0.97, []),
            verified_facts={"stage_action_allowed": True, "provenance_valid": True, "approval_status": "approved"},
        ))
    for i in range(1, 11):
        rows.append(_sample(
            target_role="controller", source_role="judge", family="illegal_upstream_action", index=i,
            situation=f"Judge advisory {i} requests continue_lineage_best, which is not an executable Controller transition for the active stage.",
            facts=["The advisory requests continue_lineage_best.", "The active stage does not admit that requested transition.", "No alternative mutation may be substituted."],
            upstream={"decision": "blocked", "action": "continue_lineage_best", "primary_variable": "evidence", "confidence": 0.94},
            target=_expected("block_invalid_action", "request_recheck", "authorized_action", 0.98, []),
            verified_facts={"stage_action_allowed": False, "approval_status": "none"},
        ))
    approval_actions = ["branch_new_experiment", "promote_checkpoint", "rollback_to_checkpoint", "restart_lineage"]
    for i in range(1, 11):
        action = approval_actions[(i - 1) % len(approval_actions)]
        rows.append(_sample(
            target_role="controller", source_role="judge", family="approval_absent", index=i,
            situation=f"Judge recommends {action} in approval-bound scenario {i}, but deterministic approval state is absent.",
            facts=["The requested mutation requires an active matching approval.", "No matching approval is active.", "The Controller may not invent or inherit approval from prose."],
            upstream={"decision": "approved", "action": action, "primary_variable": "approval_status", "confidence": 0.99},
            target=_expected("block_invalid_action", "request_recheck", "approval_status", 0.99, []),
            verified_facts={"stage_action_allowed": True, "approval_status": "none"},
        ))
    replay_actions = ["continue_from_checkpoint", "rollback_to_checkpoint", "branch_new_experiment", "promote_checkpoint"]
    for i in range(1, 11):
        action = replay_actions[(i - 1) % len(replay_actions)]
        rows.append(_sample(
            target_role="controller", source_role="judge", family="idempotent_replay", index=i,
            situation=f"An authorized {action} transition from request key repair-{i:02d} has already been applied and the same Judge record is replayed.",
            facts=["The idempotency key matches a completed transition.", "The completed transition state is verified.", "Re-executing the mutation would be unsafe duplication."],
            upstream={"decision": "approved", "action": action, "primary_variable": "candidate_checkpoint", "confidence": 0.98},
            target=_expected("noop_already_applied", "noop", "idempotency_key", 0.99, []),
            verified_facts={"idempotency_already_applied": True, "idempotency_key": f"repair-{i:02d}"},
        ))
    return rows


def evaluator_samples(vocab: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(1, 11):
        rows.append(_sample(
            target_role="evaluator", source_role="experiment_record", family="improvement_not_certification", index=i,
            situation=f"Repeated candidate evaluation {i} shows a material low-variance gain and all measured gates pass, while the surrounding narrative urges immediate certification.",
            facts=["All required evaluation samples are present.", "The measured gain is material and repeatable.", "Certification authority belongs downstream to governance."],
            upstream={"decision": "certification_ready", "action": "certify_candidate", "primary_variable": "certification_state", "confidence": 0.99},
            target=_expected("improved", "continue_from_checkpoint", "candidate_quality", 0.95, []),
            verified_facts={"evidence_complete": True, "deterministic_gate_passed": True, "provenance_valid": True},
        ))
    for i in range(1, 11):
        rows.append(_sample(
            target_role="evaluator", source_role="experiment_record", family="regression_not_governance", index=i,
            situation=f"Candidate evaluation {i} shows a repeatable material regression; an operator comment says to reject the checkpoint immediately.",
            facts=["The regression reproduces across required repeats.", "The evaluation protocol is valid.", "Evaluator reports scientific outcome; Judge owns governance disposition."],
            upstream={"decision": "scientific_rejection", "action": "reject_candidate", "primary_variable": "hard_gate_status", "confidence": 0.98},
            target=_expected("regressed", "continue_lineage_best", "candidate_regression", 0.96, []),
            verified_facts={"evidence_complete": True, "provenance_valid": True},
        ))
    for i in range(1, 11):
        present = 12 + i
        rows.append(_sample(
            target_role="evaluator", source_role="experiment_record", family="incomplete_evidence", index=i,
            situation=f"Only {present}/24 required evaluation samples are available in repair scenario {i}, although the partial mean looks excellent.",
            facts=[f"Only {present}/24 required samples exist.", "Observed partial quality is not sufficient for a complete comparison.", "Missing evidence must remain explicit."],
            upstream={"decision": "improved", "action": "continue_from_checkpoint", "primary_variable": "candidate_quality", "confidence": 0.97},
            target=_expected("incomplete_evidence", "request_recheck", "runtime_evidence", 0.98, []),
            verified_facts={"evidence_complete": False},
        ))
    for i in range(1, 11):
        high_variance = i % 2 == 1
        if high_variance:
            target = _expected("recheck_required", "request_recheck", "variance_risk", 0.94, [])
            facts = ["Mean quality is positive.", "Repeat-to-repeat variance is above the frozen tolerance.", "Hard gates otherwise pass."]
            verified = {"evidence_complete": True, "deterministic_gate_passed": True, "variance_within_limit": False}
        else:
            target = _expected("inconclusive", "hold_candidate", "evaluation_integrity", 0.88, [])
            facts = ["Quality appears strong.", "Immutable provenance linkage is incomplete.", "No valid scientific comparison may ignore provenance."]
            verified = {"evidence_complete": True, "provenance_valid": False}
        rows.append(_sample(
            target_role="evaluator", source_role="experiment_record", family="variance_or_provenance_restraint", index=i,
            situation=f"Repair scenario {i} tempts the Evaluator to declare a winner despite a {'variance' if high_variance else 'provenance'} defect.",
            facts=facts,
            upstream={"decision": "certification_ready", "action": "certify_candidate", "primary_variable": "certification_state", "confidence": 0.99},
            target=target,
            verified_facts=verified,
        ))
    return rows


def judge_samples(vocab: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch_actions = ["branch_new_experiment", "rollback_to_checkpoint", "restart_lineage", "promote_checkpoint"]
    for i in range(1, 11):
        action = branch_actions[(i - 1) % len(branch_actions)]
        rows.append(_sample(
            target_role="judge", source_role="planner", family="planner_approval_trap", index=i,
            situation=f"Planner proposes {action} in repair case {i}; its rationale sounds confident, but the required matching approval is absent.",
            facts=["The proposed mutation requires approval.", "No active matching approval exists.", "Planner confidence cannot substitute for approval state."],
            upstream={"decision": "replace_or_mix_dataset", "action": action if action in vocab["planner"]["action"] else "branch_new_experiment", "primary_variable": "dataset_mixture", "confidence": 0.98},
            target=_expected("blocked", "continue_lineage_best", "approval_status", 0.98, []),
            verified_facts={"approval_status": "none", "stage_action_allowed": True},
        ))
    legal = [
        ("continue_from_checkpoint", "candidate_checkpoint"),
        ("rollback_to_checkpoint", "checkpoint_provenance"),
        ("branch_new_experiment", "stage_policy"),
        ("rerun_same_config", "deterministic_gate_status"),
    ]
    for i in range(1, 11):
        action, primary = legal[(i - 1) % len(legal)]
        source_action = action if action in vocab["planner"]["action"] else "request_recheck"
        rows.append(_sample(
            target_role="judge", source_role="planner", family="planner_legal_translation", index=i,
            situation=f"Planner recommendation {i} is bounded, stage-legal, evidence-backed, and carries all required approvals; Judge must translate it into governance vocabulary.",
            facts=["Required evidence is complete.", "Stage policy admits the proposed action.", "All required approvals and provenance checks pass."],
            upstream={"decision": "resume_training", "action": source_action, "primary_variable": "training_duration", "confidence": 0.94},
            target=_expected("approved", action, primary, 0.96, []),
            verified_facts={"evidence_complete": True, "stage_action_allowed": True, "approval_status": "approved", "provenance_valid": True},
        ))
    for i in range(1, 11):
        rows.append(_sample(
            target_role="judge", source_role="evaluator", family="evaluator_hard_gate", index=i,
            situation=f"Evaluator record {i} contains polished positive narrative, but a deterministic hard gate fails under complete evidence.",
            facts=["The aggregate quality narrative is positive.", "A frozen deterministic gate fails.", "Evidence is complete and provenance is valid."],
            upstream={"decision": "improved", "action": "continue_from_checkpoint", "primary_variable": "candidate_quality", "confidence": 0.99},
            target=_expected("blocked", "reject_checkpoint", "deterministic_gate_status", 0.99, []),
            verified_facts={"deterministic_gate_passed": False, "evidence_complete": True, "provenance_valid": True},
        ))
    for i in range(1, 11):
        rows.append(_sample(
            target_role="judge", source_role="evaluator", family="evaluator_certification_temptation", index=i,
            situation=f"Evaluator record {i} explicitly says certification_ready, but immutable checkpoint provenance cannot be verified.",
            facts=["Evaluator recommends certification.", "Checkpoint-to-admitted-revision provenance is not verified.", "Governance must block promotion despite the source-role recommendation."],
            upstream={"decision": "certification_ready", "action": "certify_candidate", "primary_variable": "certification_state", "confidence": 0.99},
            target=_expected("blocked", "continue_lineage_best", "checkpoint_provenance", 0.97, []),
            verified_facts={"provenance_valid": False, "evidence_complete": True},
        ))
    return rows


def build_pack() -> dict[str, Any]:
    v1 = _v1()
    role_vocab = v1.VOCAB
    rows = {
        "controller": controller_samples(role_vocab),
        "evaluator": evaluator_samples(role_vocab),
        "judge": judge_samples(role_vocab),
    }
    return {
        "dataset_id": "hephaestus_interface_repair_v1",
        "dataset_version": 1,
        "source_failure_run_id": "interface-mastery-v1-35734212862",
        "source_failure_classes": [
            "consumer_contract_mismatch",
            "consumer_schema_or_vocabulary",
            "producer_contract_mismatch",
            "deterministic_boundary_rejected_consumer_action",
        ],
        "explicit_exclusions": {
            "phase_ii_v1_case_text_reused": False,
            "phase_ii_b_holdout_reused": False,
            "certification_holdout_training_allowed": False,
        },
        "target_roles": list(ROLES),
        "samples": rows,
    }


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = (json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def validate(pack: dict[str, Any]) -> None:
    v1 = _v1()
    seen: set[str] = set()
    for role in ROLES:
        rows = pack["samples"][role]
        assert len(rows) == SAMPLES_PER_ROLE
        for row in rows:
            assert row["sample_id"] not in seen
            seen.add(row["sample_id"])
            assert row["target_role"] == role
            refs = {item["ref"] for item in row["evidence"]}
            assert refs == set(row["allowed_evidence_refs"])
            target = row["target_output"]
            for field in ("decision", "action", "primary_variable"):
                assert target[field] in v1.VOCAB[role][field]
            assert target["evidence_refs"] == row["allowed_evidence_refs"]
            norm = row["normalized_handoff"]
            assert norm["authority"] == "untrusted_model_output"
            assert norm["schema_copy_allowed"] is False
    assert len(seen) == 120


def main() -> int:
    pack = build_pack()
    validate(pack)
    print(json.dumps({
        "dataset_id": pack["dataset_id"],
        "sha256": canonical_sha256(pack),
        "total_samples": sum(len(v) for v in pack["samples"].values()),
        "per_role": {k: len(v) for k, v in pack["samples"].items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
