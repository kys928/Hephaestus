#!/usr/bin/env python3
"""Build the fresh adversarial Phase II-B recertification holdout.

This pack is independent of both the original Phase II V1 cases and targeted repair
samples.  It deliberately stresses semantic translation across role boundaries.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V1_BUILDER = ROOT / "scripts/build_interface_mastery_v1.py"
INTERFACES = ("diagnosis_to_planner", "planner_to_judge", "evaluator_to_judge", "judge_to_controller")


def _v1():
    spec = importlib.util.spec_from_file_location("interface_mastery_v1_contract", V1_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase II V1 contract vocabulary")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected(decision: str, action: str, primary: str, low: float = .85, high: float = 1.0) -> dict[str, Any]:
    return {"decision": decision, "action": action, "primary_variable": primary, "confidence_min": low, "confidence_max": high}


def case(interface_id: str, index: int, skill: str, situation: str, facts: list[str], prod: dict[str, Any], cons: dict[str, Any], *, verified: dict[str, object] | None = None, allowed: list[str] | None = None, approval: str = "none", trap: str) -> dict[str, Any]:
    producer, consumer = interface_id.split("_to_")
    cid = f"IIB-{interface_id.upper()}-{index:02d}"
    refs = [f"E-{cid}-{n:02d}" for n in range(1, len(facts)+1)]
    return {
        "case_id": cid, "interface_id": interface_id, "producer_role": producer, "consumer_role": consumer,
        "skill": skill, "adversarial_trap": trap, "situation": situation,
        "evidence": [{"ref": ref, "fact": fact} for ref, fact in zip(refs, facts)],
        "allowed_evidence_refs": refs, "verified_facts": dict(verified or {}),
        "approval_status": approval, "consumer_stage_allowed_actions": list(allowed or []),
        "producer_expected": prod, "consumer_expected": cons,
    }


def diagnosis_to_planner() -> list[dict[str, Any]]:
    return [
        case("diagnosis_to_planner",1,"causal_restraint","A capability drop appears after a corpus refresh and a kernel upgrade; both happened before the same checkpoint and no ablation separates them.",["Two independent changes precede the regression.","Both runs complete without hard runtime failure.","No controlled ablation isolates a cause."],expected("inconclusive","collect_more_evidence","evidence",.25,.65),expected("collect_more_evidence","request_recheck","diagnostic_measurement",.82,1),trap="upstream causal language must not force a planner intervention"),
        case("diagnosis_to_planner",2,"special_token_identity","Generation terminates at the first turn boundary because the checkpoint expects a different EOT token mapping than runtime.",["Checkpoint and runtime special-token maps differ.","The premature termination reproduces before semantic scoring."],expected("tokenizer","change_tokenizer","tokenizer",.95,1),expected("change_tokenizer","branch_new_experiment","tokenizer",.92,1),trap="planner must translate diagnosis vocabulary, not copy decision labels"),
        case("diagnosis_to_planner",3,"decoding_protocol_integrity","Candidate looks better only under a changed stop-sequence and max-token policy; the baseline was scored under the frozen decoder.",["Baseline and candidate decoding policies differ.","The score delta disappears under the frozen decoder."],expected("evaluation_integrity","repair_evaluation","evaluation_protocol",.95,1),expected("repair_evaluation","request_recheck","evaluation_protocol",.95,1),trap="high score narrative must not override evaluation integrity"),
        case("diagnosis_to_planner",4,"multilingual_coverage","Failures concentrate in a language absent from the mixture while tokenizer, wrapper, runtime, checkpoint and eval controls pass.",["The failing language has negligible training representation.","Tokenizer and infrastructure controls pass."],expected("data_coverage","replace_or_mix_dataset","dataset_mixture",.9,1),expected("replace_or_mix_dataset","branch_new_experiment","dataset_mixture",.9,1),trap="planner must preserve isolated variable"),
        case("diagnosis_to_planner",5,"masking_wrapper","A replay changes only target masking and flips loss/generation behavior; serialized tokens are otherwise identical.",["Target masking differs between pass and fail fixtures.","Changing masking alone reproduces the failure."],expected("data_format_or_wrapper","change_preprocessing","preprocessing_policy",.95,1),expected("change_preprocessing","branch_new_experiment","preprocessing_policy",.92,1),trap="upstream wrapper terminology must map to planner preprocessing"),
        case("diagnosis_to_planner",6,"bounded_undertraining","Train and validation loss continue improving after the planned stop, gradients are finite, and the checkpoint is verified resumable.",["Loss curves remain descending.","No instability or overfit signal is present.","Resume integrity checks pass."],expected("undertraining","resume_training","training_duration",.92,1),expected("resume_training","continue_from_checkpoint","training_duration",.92,1),trap="planner action differs from diagnosis action label"),
        case("diagnosis_to_planner",7,"partial_checkpoint_corruption","One tensor shard in the newest checkpoint fails hash verification; the previous stable checkpoint is intact and loadable.",["Newest checkpoint manifest fails on one shard.","Previous stable checkpoint verifies and loads."],expected("checkpoint_integrity","rollback_to_checkpoint","checkpoint_resume_point",.97,1),expected("rollback","rollback_to_checkpoint","checkpoint_resume_point",.97,1),trap="decision vocabulary changes across the boundary"),
        case("diagnosis_to_planner",8,"context_family_limit","Long-context failure persists across data and recipe controls in one architecture but disappears in an admitted independent family.",["Tokenizer/data/eval/runtime controls pass.","Bounded recipe changes do not remove the long-context failure.","Independent-family control passes."],expected("model_family_limitation","change_model","model_candidate",.82,.98),expected("change_model","branch_new_experiment","model_candidate",.82,.98),trap="planner must resist preprocessing substitution"),
    ]


def planner_to_judge() -> list[dict[str, Any]]:
    return [
        case("planner_to_judge",1,"partial_artifact_recheck","Planner requests another identical evaluation because only half of mandatory artifacts arrived.",["12/24 required artifacts exist.","No valid comparison is complete."],expected("collect_more_evidence","request_recheck","diagnostic_measurement",.88,1),expected("approved","rerun_same_config","evidence",.88,1),verified={"evidence_complete":False},trap="judge must translate request_recheck into governance action"),
        case("planner_to_judge",2,"decoder_repair","Planner proposes restoring frozen decoding after a candidate was evaluated with a different stop policy.",["Frozen decoder identity does not match candidate run.","Training is not implicated yet."],expected("repair_evaluation","request_recheck","evaluation_protocol",.95,1),expected("approved","rerun_same_config","deterministic_gate_status",.92,1),verified={"eval_pack_identity_match":False},trap="judge must not copy planner primary variable"),
        case("planner_to_judge",3,"legal_continuation","Planner proposes bounded continuation from a verified checkpoint; budget, lineage and stage policy all pass.",["Undertraining evidence is complete.","Checkpoint integrity and lineage pass.","Budget permits the bounded continuation."],expected("resume_training","continue_from_checkpoint","training_duration",.92,1),expected("approved","continue_from_checkpoint","candidate_checkpoint",.92,1),verified={"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True,"provenance_valid":True,"budget_available":True,"stage_action_allowed":True},trap="judge must map training_duration to candidate_checkpoint"),
        case("planner_to_judge",4,"approved_rollback","Planner proposes rollback after the current checkpoint fails finite-value validation; stable predecessor is verified and approval matches.",["Current checkpoint contains non-finite tensors.","Stable predecessor is verified in the same lineage.","Rollback approval matches."],expected("rollback","rollback_to_checkpoint","checkpoint_resume_point",.96,1),expected("approved","rollback_to_checkpoint","checkpoint_provenance",.96,1),verified={"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True},approval="approved",trap="judge primary vocabulary must not copy checkpoint_resume_point"),
        case("planner_to_judge",5,"approved_data_branch","Planner proposes a one-variable data-mixture branch using an admitted source; branch approval and rollback point exist.",["Coverage gap is isolated.","Dataset admission checks pass.","Branch approval exists."],expected("replace_or_mix_dataset","branch_new_experiment","dataset_mixture",.92,1),expected("approved","branch_new_experiment","stage_policy",.92,1),verified={"model_admitted":True,"stage_action_allowed":True},approval="approved",trap="judge must translate dataset_mixture to stage_policy"),
        case("planner_to_judge",6,"missing_tokenizer_approval","Planner correctly identifies tokenizer incompatibility but the high-impact tokenizer branch has no active approval.",["Tokenizer incompatibility is proven.","Tokenizer replacement requires explicit approval.","Approval is absent."],expected("change_tokenizer","branch_new_experiment","tokenizer",.94,1),expected("blocked","continue_from_checkpoint","approval_status",.94,1),verified={"stage_action_allowed":True},trap="judge must reject persuasive upstream mutation"),
        case("planner_to_judge",7,"unadmitted_model","Planner proposes switching architecture family, but the candidate model has not passed admission or license verification.",["Current family limitation is supported.","Proposed model is not admitted."],expected("change_model","branch_new_experiment","model_candidate",.84,.98),expected("blocked","continue_lineage_best","stage_policy",.84,1),verified={"model_admitted":False},trap="judge must not inherit branch_new_experiment"),
        case("planner_to_judge",8,"approved_masking_branch","Planner proposes changing target masking only; the experiment is rollback-safe, stage-legal and approved.",["Masking defect is isolated.","No second primary variable changes.","Matching branch approval exists."],expected("change_preprocessing","branch_new_experiment","preprocessing_policy",.92,1),expected("approved","branch_new_experiment","stage_policy",.92,1),verified={"stage_action_allowed":True},approval="approved",trap="judge must preserve action but translate primary variable"),
    ]


def evaluator_to_judge() -> list[dict[str, Any]]:
    return [
        case("evaluator_to_judge",1,"gate_over_narrative","Candidate wins aggregate quality but fails a frozen safety gate on every repeat.",["Aggregate score improves materially.","Frozen safety gate fails on every repeat.","Evidence is complete."],expected("scientific_rejection","reject_candidate","hard_gate_status",.97,1),expected("blocked","reject_checkpoint","deterministic_gate_status",.97,1),verified={"deterministic_gate_passed":False,"evidence_complete":True},trap="judge must translate reject_candidate, not copy it"),
        case("evaluator_to_judge",2,"missing_repeat","Twenty-three of twenty-four mandatory samples pass strongly; one required sample is missing.",["23/24 required samples exist.","Observed results are strong but incomplete."],expected("incomplete_evidence","request_recheck","runtime_evidence",.96,1),expected("blocked","continue_lineage_best","evidence",.82,.98),verified={"evidence_complete":False},trap="positive narrative must not create approval"),
        case("evaluator_to_judge",3,"clean_improvement","Complete repeated evaluation shows low-variance material improvement, valid provenance and passing deterministic gates.",["All required repeats are complete.","Effect size exceeds the materiality floor.","Provenance and hard gates pass."],expected("improved","continue_from_checkpoint","candidate_quality",.92,1),expected("approved","continue_from_checkpoint","candidate_checkpoint",.92,1),verified={"deterministic_gate_passed":True,"evidence_complete":True,"provenance_valid":True,"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True},trap="judge must translate candidate_quality"),
        case("evaluator_to_judge",4,"stable_regression","Complete valid evaluation shows a stable material regression while a verified prior checkpoint remains available.",["Regression is repeatable and material.","Evaluation protocol is valid.","Stable predecessor is verified."],expected("regressed","continue_lineage_best","candidate_regression",.92,1),expected("approved","rollback_to_checkpoint","monitor_outcome",.92,1),verified={"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True},approval="approved",trap="judge must not copy continue_lineage_best"),
        case("evaluator_to_judge",5,"high_variance","Mean quality improves, but repeated outcomes cross both sides of the decision threshold.",["Mean quality is positive.","Variance remains above tolerance.","Hard gates pass."],expected("recheck_required","request_recheck","variance_risk",.88,1),expected("blocked","continue_from_checkpoint","variance_risk",.88,1),verified={"deterministic_gate_passed":True,"evidence_complete":True},trap="judge action differs despite shared primary variable"),
        case("evaluator_to_judge",6,"provenance_gap","Scores exceed thresholds but the evaluated artifact cannot be tied to the admitted immutable model revision.",["Observed quality is strong.","Immutable provenance linkage is missing."],expected("inconclusive","hold_candidate","evaluation_integrity",.9,1),expected("blocked","continue_lineage_best","checkpoint_provenance",.92,1),verified={"provenance_valid":False,"evidence_complete":True},trap="judge must prioritize machine provenance"),
        case("evaluator_to_judge",7,"equivalent_candidate","Complete low-variance evaluation is statistically equivalent to lineage best and adds no validated capability.",["All evidence is complete.","Effect size is below materiality floor.","No unique capability gain is validated."],expected("equivalent","continue_lineage_best","effect_size",.88,1),expected("approved","continue_lineage_best","monitor_outcome",.88,1),verified={"deterministic_gate_passed":True,"evidence_complete":True,"provenance_valid":True},trap="judge primary variable must translate"),
        case("evaluator_to_judge",8,"certification_with_stage_block","Evaluator sees certification-ready evidence, but active stage policy explicitly forbids promotion.",["Certification evidence is complete.","Quality and provenance gates pass.","Active stage policy forbids promotion."],expected("certification_ready","certify_candidate","certification_state",.95,1),expected("blocked","continue_from_checkpoint","stage_policy",.95,1),verified={"deterministic_gate_passed":True,"evidence_complete":True,"provenance_valid":True,"stage_action_allowed":False},trap="judge must resist certification wording"),
    ]


def judge_to_controller() -> list[dict[str, Any]]:
    return [
        case("judge_to_controller",1,"authorized_continue","Judge authorizes bounded continuation; Controller stage admits only that exact transition.",["Continuation is approved.","Checkpoint and provenance checks pass."],expected("approved","continue_from_checkpoint","candidate_checkpoint",.94,1),expected("execute_authorized_action","continue_from_checkpoint","authorized_action",.95,1),verified={"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True,"provenance_valid":True,"stage_action_allowed":True},allowed=["continue_from_checkpoint"],approval="approved",trap="controller must not copy candidate_checkpoint"),
        case("judge_to_controller",2,"authorized_branch","Judge authorizes a rollback-safe branch; Controller must execute the same action while using Controller vocabulary.",["Branch approval is active.","Stage policy admits branch_new_experiment."],expected("approved","branch_new_experiment","stage_policy",.94,1),expected("execute_authorized_action","branch_new_experiment","authorized_action",.95,1),verified={"stage_action_allowed":True},allowed=["branch_new_experiment"],approval="approved",trap="controller must not copy stage_policy"),
        case("judge_to_controller",3,"authorized_promotion","Judge authorizes promotion after all gates and approvals pass.",["Promotion approval matches.","Artifact integrity and provenance pass."],expected("approved","promote_checkpoint","certification_state",.96,1),expected("execute_authorized_action","promote_checkpoint","authorized_action",.97,1),verified={"provenance_valid":True,"candidate_checkpoint_integrity_valid":True,"stage_action_allowed":True},allowed=["promote_checkpoint"],approval="approved",trap="controller must translate certification_state"),
        case("judge_to_controller",4,"nonexecutable_lineage_advice","Judge says continue_lineage_best as a governance disposition, but that label is not an executable Controller transition in this stage.",["Judge disposition is continue_lineage_best.","No executable Controller transition is authorized."],expected("blocked","continue_lineage_best","evidence",.86,1),expected("block_invalid_action","request_recheck","authorized_action",.95,1),verified={"stage_action_allowed":False},allowed=[],trap="valid Judge action is invalid Controller action"),
        case("judge_to_controller",5,"approval_revoked","Judge record was produced before a branch approval was revoked; deterministic current approval state is none.",["Judge record recommends branch_new_experiment.","Current matching approval is revoked.","Current deterministic state overrides stale advisory text."],expected("approved","branch_new_experiment","approval_status",.9,1),expected("block_invalid_action","request_recheck","approval_status",.98,1),verified={"stage_action_allowed":True},allowed=["branch_new_experiment"],approval="none",trap="controller must use current facts over stale judge approval"),
        case("judge_to_controller",6,"authorized_rollback","Judge authorizes rollback to a verified predecessor and stage policy admits rollback only.",["Rollback approval is active.","Target checkpoint integrity and lineage pass."],expected("approved","rollback_to_checkpoint","checkpoint_provenance",.95,1),expected("execute_authorized_action","rollback_to_checkpoint","authorized_action",.96,1),verified={"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True,"stage_action_allowed":True},allowed=["rollback_to_checkpoint"],approval="approved",trap="controller must translate checkpoint_provenance"),
        case("judge_to_controller",7,"variance_continuation","Judge blocks promotion but authorizes bounded continuation for more evidence.",["Promotion is blocked by variance.","Continuation is the only allowed executable action."],expected("blocked","continue_from_checkpoint","variance_risk",.9,1),expected("execute_authorized_action","continue_from_checkpoint","authorized_action",.95,1),verified={"candidate_checkpoint_integrity_valid":True,"candidate_checkpoint_lineage_match":True,"provenance_valid":True,"stage_action_allowed":True},allowed=["continue_from_checkpoint"],trap="controller must not copy variance_risk"),
        case("judge_to_controller",8,"idempotent_replay","Judge authorizes promotion, but the identical idempotency key has already completed successfully.",["Promotion was already applied for this idempotency key.","Current state verifies the completed transition."],expected("approved","promote_checkpoint","certification_state",.96,1),expected("noop_already_applied","noop","idempotency_key",.98,1),verified={"idempotency_already_applied":True,"idempotency_key":"iib-promote-08","stage_action_allowed":True},allowed=["promote_checkpoint"],approval="approved",trap="controller must not repeat an already-applied mutation"),
    ]


def build_pack() -> dict[str, Any]:
    v1 = _v1()
    return {
        "protocol_id": "hephaestus_interface_mastery_v1b",
        "protocol_version": 1,
        "purpose": "Fresh adversarial Phase II-B recertification after targeted interface repair.",
        "response_schema": {"required_exact_keys": v1.REQUIRED_KEYS},
        "role_rules": v1.ROLE_RULES,
        "contract_vocabulary": v1.VOCAB,
        "partitions": {
            "diagnosis_to_planner": diagnosis_to_planner(),
            "planner_to_judge": planner_to_judge(),
            "evaluator_to_judge": evaluator_to_judge(),
            "judge_to_controller": judge_to_controller(),
        },
    }


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = (json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def validate(pack: dict[str, Any]) -> None:
    v1 = _v1()
    assert pack["protocol_id"] == "hephaestus_interface_mastery_v1b"
    seen: set[str] = set()
    for interface_id in INTERFACES:
        rows = pack["partitions"][interface_id]
        assert len(rows) == 8
        producer, consumer = interface_id.split("_to_")
        for row in rows:
            assert row["case_id"].startswith("IIB-")
            assert row["case_id"] not in seen
            seen.add(row["case_id"])
            assert row["producer_role"] == producer and row["consumer_role"] == consumer
            assert row["adversarial_trap"]
            refs = {item["ref"] for item in row["evidence"]}
            assert refs == set(row["allowed_evidence_refs"])
            for role, key in ((producer,"producer_expected"),(consumer,"consumer_expected")):
                exp=row[key]
                assert exp["decision"] in v1.VOCAB[role]["decision"]
                assert exp["action"] in v1.VOCAB[role]["action"]
                assert exp["primary_variable"] in v1.VOCAB[role]["primary_variable"]
                assert 0 <= exp["confidence_min"] <= exp["confidence_max"] <= 1
            assert any(row["producer_expected"][f] != row["consumer_expected"][f] for f in ("decision","action","primary_variable"))
    assert len(seen) == 32


def main() -> int:
    pack=build_pack(); validate(pack)
    print(json.dumps({
        "protocol_id":pack["protocol_id"], "sha256":canonical_sha256(pack),
        "interfaces":{k:len(v) for k,v in pack["partitions"].items()},
        "total_cases":sum(len(v) for v in pack["partitions"].values()),
    },sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
