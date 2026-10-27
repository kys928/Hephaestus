"""Five actual CPU *research-role* advisory passes on verified PEFT evidence.

Uses the already present and pinned 135M pretrained CPU causal language model.
A single role-conditioned CPU research model is NOT five distinct frozen CUDA
role models, is not certified, and can NEVER issue acquisition/action approvals.
"""
from __future__ import annotations
import json
from pathlib import Path

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.development.cpu_cycle import digest, canonical, save_json
from hephaestus.development.existing_peft_bridge import verify_peft_cycle
from hephaestus.llm.cpu_causal_research import CpuCausalProposalScorer, CANDIDATES
from hephaestus.llm.role_contract import RoleTask, RoleEvidence
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json


RUN = "cpu-peft-development"
LINEAGE = "local-development-only"
STAGE = "smoke_test"


def run_cpu_role_advisories(root: Path, *, cache_dir: Path) -> dict:
    replay = verify_peft_cycle(root)
    if not replay["valid"]:
        raise ValueError("only verified completed local PEFT evidence is admissible")
    manifest = json.loads((root / "peft_cycle_manifest.json").read_text())
    verdict = json.loads((root / "judge_verdict.json").read_text())
    if (manifest["schema"] != "hephaestus-existing-peft-dev-bridge.v1"
            or not manifest["frozen_five_role_production_models_not_loaded"]
            or manifest["original_governed_45_stage_progress"] != "11/45_unchanged"
            or verdict["45_stage_campaign_advanced"]):
        raise PermissionError("CPU role study may not reinterpret production role/campaign authority")

    artifacts = FileSystemArtifactStore(root / "role_research_artifacts")
    # The role model reads an immutable validated source report created from
    # real gradients/generation/Judge, not synthetic citations or fixture text.
    source = {
        "schema": "cpu-peft-observed-result-for-role-proposals.v1",
        "source_proposal_ref": digest((root / "local_development_proposal.json").read_bytes()),
        "observed_manifest_ref": digest((root / "peft_cycle_manifest.json").read_bytes()),
        "observed_judge_ref": digest((root / "judge_verdict.json").read_bytes()),
        "actual_lora_checkpoint_manifest": manifest["actual_peft_checkpoint_manifest"],
        "actual_target_correctness_before": verdict["primary_before"],
        "actual_target_correctness_after": verdict["primary_after"],
        "new_hard_frozen_regression_count": len(verdict["new_hard_regression_failures"]),
        "scientific_approval": False, "model_promotion": False,
        "source_kind": "actual_local_generated_data_experiment",
    }
    record = artifacts.put_bytes(canonical(source), media_type="application/json")
    ref = record.artifact_ref
    registry = EvidenceRegistry({
        ref: EvidenceRecord(
            ref=ref, kind="verified_noncertified_cpu_peft_evidence",
            run_id=RUN, lineage_id=LINEAGE,
            sha256=ref.removeprefix("sha256:"),
            observed_sha256=ref.removeprefix("sha256:"),
            facts={"approval_valid": False, "evidence_complete": False,
                   "recheck_satisfied": False, "stage_action_allowed": False},
        )
    })
    scorer = CpuCausalProposalScorer.load(artifacts, cache_dir=cache_dir)
    findings = []
    for role in CANDIDATES:
        situation = (
            "A real bounded original-data CPU LoRA training run has completed "
            "using TransformersTrainingLifecycleService with real generated "
            "outputs on an unchanged precommitted eval plan. Its scientific "
            f"target correctness was {verdict['primary_before']} of 20 before and "
            f"{verdict['primary_after']} of 20 after, and there were "
            f"{len(verdict['new_hard_regression_failures'])} newly failed general "
            "hard checks. This research run has no acquisition or production "
            "approval and cannot promote any checkpoint. Recommend next review."
        )
        task = RoleTask(
            role=role, situation=situation,
            evidence=(RoleEvidence(ref, (
                "Verified real PEFT experiment; primary correctness "
                f"{verdict['primary_before']} to {verdict['primary_after']} "
                "on 20 original-development examples; production approvals absent."
            )),),
            run_id=RUN, lineage_id=LINEAGE, stage_name=STAGE,
        )
        context = DeterministicRoleContext(
            RUN, LINEAGE, STAGE, approval_status="none",
            verified_facts={"approval_valid": False, "stage_action_allowed": False},
        )
        observation = scorer.propose(task, registry, context)
        if (observation["action_executed"] or not observation["advisory_only"]
                or observation["approval_granted"]
                or observation["production_role_certified"]
                or observation["scientific_quality_certified"]):
            raise PermissionError("CPU role advisory attempted to claim production authority")
        saved = read_verified_json(artifacts, observation["inference_receipt_ref"])
        if saved["selected_candidate"] != observation["selected_candidate"]:
            raise ValueError("model-selected proposal differs from SHA-stored inference")
        findings.append({
            "role":role, "selected":observation["selected_candidate"],
            "inference_receipt_ref":observation["inference_receipt_ref"],
            "model_id":observation["identity"]["model_id"],
            "model_revision":observation["identity"]["revision"],
            "model_weights_file_sha256":observation["identity"]["model_weights_file_sha256"],
            "advisory_only":True,
        })
    if len(findings)!=5 or len({x["role"] for x in findings})!=5:
        raise ValueError("missing actual model-conditioned research roles")
    report = {
        "schema":"hephaestus-five-cpu-research-role-after-real-peft.v1",
        "real_peft_manifest_sha256":source["observed_manifest_ref"],
        "model_family":"one_pinned_pretrained_135m_cpu_causal_lm",
        "distinct_frozen_five_cuda_models_loaded":False,
        "frozen_five_role_mastery_contract_satisfied":False,
        "all_five_role_slots_model_conditioned":True,
        "approval_issued":False,"model_promoted":False,
        "production_staged_orchestrator_executed":False,
        "original_45_campaign_advanced":False,
        "role_outputs":findings,
    }
    ref=save_json(root/"cpu_role_advisory_report.json",report)
    return {"status":"completed_cpu_research_advisory_only",
            "role_count":len(findings),"report_sha256":ref,
            "production_certification":False,"45_campaign_advanced":False}


def verify_cpu_role_advisories(root:Path)->dict:
    verify_peft_cycle(root)
    report=json.loads((root/"cpu_role_advisory_report.json").read_text())
    if (report["schema"]!="hephaestus-five-cpu-research-role-after-real-peft.v1"
            or report["real_peft_manifest_sha256"]!=digest((root/"peft_cycle_manifest.json").read_bytes())
            or report["approval_issued"] or report["model_promoted"]
            or report["production_staged_orchestrator_executed"]
            or report["original_45_campaign_advanced"]
            or len(report["role_outputs"])!=5):
        raise ValueError("CPU role result identity or production authority mutated")
    store=FileSystemArtifactStore(root/"role_research_artifacts")
    for role in report["role_outputs"]:
        src=read_verified_json(store,role["inference_receipt_ref"])
        if src["role"]!=role["role"] or src["selected_candidate"]!=role["selected"]:
            raise ValueError("CPU role receipt/hash mismatch")
        if not src["advisory_only"] or src["action_executed"] or src["approval_granted"]:
            raise ValueError("model proposal attempted protected action")
    return {"schema":"hephaestus-cpu-research-role-replay.v1",
            "valid":True,"verified_real_model_role_receipts":5,
            "production_five_role_inference":False,
            "original_45_campaign_advanced":False}
