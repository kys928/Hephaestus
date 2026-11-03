"""Decision-first authentic Prometheus research, not a preselected Granite repair.

The scientist is Hephaestus's REAL frozen Diagnostician and Planner, with their
own SHA-verified PEFT adapters and pinned model revisions on one CUDA GPU.
Only independently observed, explicitly non-comparable development aggregates
enter the model. Deterministic diagnosis and intervention admission remain
authoritative. No parent model is altered, and no training/promotion is launched.

Usage (inside an admitted, independently supervised one-GPU worker):
 python -m hephaestus.development.prometheus_scientific_decision \
    --parent-report /workspace/admitted/original-parent-dev.json \
    --d-report /workspace/admitted/historical-d-dev.json \
    --s3-endpoint https://s3api-eu-cz-1.runpod.io/ \
    --s3-bucket "$RUNPOD_NETWORK_VOLUME_ID" \
    --out /workspace/private/prometheus/decision-first \
    --adapter-cache /workspace/role-adapters \
    --hf-cache /tmp/prometheus-hf-cache

The two reports are named source inputs with independently retained SHA-256
digests. Neither is a fresh independent model assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError, _parse_json
from hephaestus.diagnosis.service import EvidenceBasedDiagnosisService
from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.llm.role_contract import RoleEvidence, RoleMasteryContract, RoleTask
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.policy.model_proposals import planner_selection
from hephaestus.providers.models.role_stack import load_certified_role_model_stack
from hephaestus.roles.diagnostician import select_model_hypothesis
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

ROOT = Path(__file__).resolve().parents[3]
PARENT_SHA = "2d1ffa37a9e14939c5c0e9c3365a5fe0a4220a37b502678a1f72f5def366378f"
D_SHA = "7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953"
PARENT_ADAPTER = "e60497854ea64655fc18be7962dec29f13091c75ea91e576726b0fa5c6eb2325"
PARENT_REVISION = "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
PARENT_ONTOLOGY = "frozen-role-mastery-evaluator.v1"
D_ONTOLOGY = "historical-semantic-invariance-evaluator.v1"
RUN = "prometheus-original-granite-science-v1"
LINEAGE = "prometheus-frozen-parent-diagnostic-only"
STAGE = "continuation_repair"
REF_PARENT = "sha256:" + PARENT_SHA
REF_D = "sha256:" + D_SHA
ALLOWED_RESEARCH_DECISIONS = frozenset({"collect_more_evidence", "repair_evaluation", "stop"})


def _verify(path: Path, expected: str) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 40_000:
        raise PinnedEvidenceError("historical aggregate source not a bounded regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PinnedEvidenceError("historical aggregate SHA-256 mismatch")
    try:
        value = _parse_json(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PinnedEvidenceError("non-UTF8 historical aggregate") from exc
    if not isinstance(value, dict):
        raise PinnedEvidenceError("historical aggregate must be JSON")
    return value


def authenticated_observations(parent: dict, d: dict) -> tuple[list[dict], tuple[RoleEvidence, ...]]:
    """Validate the *scopes* as well as source identities, without causal inference."""
    if (
        parent.get("schema") != "prometheus-original-certified-granite-dev-rescore.v1"
        or parent.get("ontology") != PARENT_ONTOLOGY
        or parent.get("original_model_revision") != PARENT_REVISION
        or parent.get("original_selected_adapter_sha256") != PARENT_ADAPTER
        or parent.get("partition") != "previously_exposed_model_selection_development"
        or parent.get("count") != 200
        or parent.get("strict_projected_decision_correct") != 200
        or parent.get("strict_raw_decision_correct") != 200
        or parent.get("original_300_case_certification_not_read") is not True
        or parent.get("research_development_not_independent_final") is not True
        or parent.get("model_weights_newly_rehashed") is not False
    ):
        raise PinnedEvidenceError("original frozen parent measurement scope invalid")
    if (
        d.get("schema") != "prometheus-granite-d-readonly-raw-rescore.v1"
        or d.get("scope") != "historical_exposed_eight_state_development_only"
        or d.get("independent_strict_scoring", {}).get("ontology") != D_ONTOLOGY
        or d.get("independent_strict_scoring", {}).get("count") != 64
        or d.get("independent_strict_scoring", {}).get("decision_correct") != 33
        or d.get("model_weights_independently_verified") is not False
        or d.get("independent_final_assessment") is not False
    ):
        raise PinnedEvidenceError("historical D observation scope invalid")
    facts = (
        RoleEvidence(REF_PARENT,
            "Original certified Granite Evaluator, exact frozen six-state role-mastery adapter: "
            "200/200 correct on the ORIGINAL EXPOSED CHECKPOINT-SELECTION development set. "
            "This does not measure held-out generalization. The original 300 certification "
            "case responses were not opened; no new model inference has been performed."),
        RoleEvidence(REF_D,
            "Different historical Granite D research adapter, DIFFERENT EIGHT-STATE semantic "
            "research ontology: 33/64 correct on exposed development. Zero of eight equivalent "
            "and zero of eight recheck_required cases correct. These are NOT directly matched "
            "or interchangeable with the certified production parent's six-state decisions."),
    )
    return [
        {"evidence_kind":"historical_observation", "source_ref":fact.ref,
         "summary":fact.fact, "historical_exposed":True,
         "confidence":0.85 if fact.ref == REF_PARENT else 0.8}
        for fact in facts
    ], facts


def prepare(parent: dict, d: dict):
    observations, facts = authenticated_observations(parent, d)
    req = DiagnosisRequest(
        request_id="prometheus-parent-uncertainty-20261103",
        run_id=RUN, lineage_id=LINEAGE, stage_name=STAGE,
        observed_failures=observations, evidence_refs=[],
        requested_by="prometheus_original_parent_research",
    )
    diagnosis = EvidenceBasedDiagnosisService().diagnose(req)
    diagnosis.metadata.update({
        "allowed_intervention_kinds": sorted(ALLOWED_RESEARCH_DECISIONS),
        "baseline_ref":REF_PARENT,
        "baseline_justification":
            "Exact original six-state development evidence; no matched challenge baseline",
        "source_ontology":PARENT_ONTOLOGY,
        "historical_D_is_non_comparable":True,
        "model_weights_independently_rehashed":False,
    })
    options = ClosedLoopExperimentPlanner().propose_interventions(diagnosis)
    if not options or any(
        proposal.intervention_kind not in ALLOWED_RESEARCH_DECISIONS
        for proposal in options
    ):
        raise PinnedEvidenceError("historical-only diagnosis admitted an invalid intervention")
    return diagnosis, list(options), facts


def _source_registry(facts: tuple[RoleEvidence, ...]) -> EvidenceRegistry:
    return EvidenceRegistry({
        fact.ref: EvidenceRecord(
            ref=fact.ref, kind="verified_historical_exposed_summary",
            run_id=RUN, lineage_id=LINEAGE, state="verified",
            sha256=fact.ref.removeprefix("sha256:"),
            observed_sha256=fact.ref.removeprefix("sha256:"),
        ) for fact in facts
    })


def _precommit(diagnosis, options, facts) -> dict:
    return {
        "schema":"prometheus-genuine-role-decision-precommit.v1",
        "scientific_objective":
            "Determine whether the original frozen six-state Granite Evaluator needs "
            "new generalization evidence before any repair experiment",
        "source_evidence_refs":[fact.ref for fact in facts],
        "parent_ontology":PARENT_ONTOLOGY,
        "historical_D_is_non_comparable":True,
        "permitted_decisions":sorted(ALLOWED_RESEARCH_DECISIONS),
        "policy_candidates":[option.to_dict() for option in options],
        "diagnosis":diagnosis.to_dict(),
        "research_role_owner":"HEPHAESTUS_CERTIFIED_DIAGNOSIS_AND_PLANNER_IF_AUTHENTIC",
        "training_updates_authorized":0,
        "independent_final_assessment_authorized":False,
        "production_activation_authorized":False,
        "model_inference_performed_at_precommit":False,
    }


def decide(parent: dict, d: dict, *, reasoner: CertifiedRoleReasoner, out: Path) -> dict:
    """Execute *real* frozen role inference, then independently admit its result."""
    diagnosis, options, facts = prepare(parent, d)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("research decision directory must be fresh, never overwrite prior receipts")
    out.mkdir(parents=True, exist_ok=True)
    precommit = _precommit(diagnosis, options, facts)
    (out/"precommit.json").write_bytes(canonical_bytes(precommit))
    registry = _source_registry(facts)
    context = DeterministicRoleContext(RUN, LINEAGE, STAGE,
        approval_status="none",
        stage_allowed_actions=("observe_state", "request_recheck"))
    task_d = RoleTask("diagnosis",
        "Assess the measured original Granite Evaluator's scientific reliability, "
        "explicit uncertainties, and the evidentiary limits of the historical "
        "nonmatching D result. Do not assert a causal training defect from those "
        "observations alone; select an interpretation permitted by the frozen role contract.",
        facts, RUN, LINEAGE, STAGE)
    diagnosis_observation = reasoner.propose(task_d, registry, context=context,
                                             max_new_tokens=768)
    verify_inference_receipt(reasoner.artifact_store,
                              diagnosis_observation.inference_receipt_ref,
                              reasoner.contract)
    reviewed_diagnosis = select_model_hypothesis(
        diagnosis, diagnosis_observation.answer, diagnosis_observation.inference_receipt_ref)
    candidates = ClosedLoopExperimentPlanner().propose_interventions(reviewed_diagnosis)
    if not candidates or any(p.intervention_kind not in ALLOWED_RESEARCH_DECISIONS for p in candidates):
        raise PinnedEvidenceError("model diagnosis escalated historical evidence into unsafe candidate")
    options_fact = RoleEvidence(
        "prometheus:admitted-options",
        "The deterministic scientific service currently offers these candidate actions: " +
        "; ".join(sorted({f"{p.intervention_kind} with {p.primary_variable}" for p in candidates})) +
        ". These are proposals, not approvals. In the absence of demonstrated "
        "production-parent generalization errors, training is not scientifically admitted.")
    reg = _source_registry(facts)
    reg = EvidenceRegistry({
        **{f.ref:EvidenceRecord(ref=f.ref,kind="verified_historical_exposed_summary",
            run_id=RUN,lineage_id=LINEAGE,sha256=f.ref[7:],observed_sha256=f.ref[7:])
            for f in facts},
        options_fact.ref:EvidenceRecord(ref=options_fact.ref,kind="deterministic_policy_candidates",
            run_id=RUN,lineage_id=LINEAGE),
    })
    planner_task = RoleTask("planner",
        "Given the operator's objective to investigate actual original Granite Evaluator "
        "generalization, decide the next SCIENTIFIC action, including a legitimate "
        "stop or request for more evidence. Only proposals already admitted by the "
        "deterministic scientific service may be executed. Do not choose a training "
        "recipe from unmatched historical D research.",
        (*facts, options_fact), RUN, LINEAGE, STAGE)
    proposed = reasoner.propose(planner_task, reg, context=context, max_new_tokens=768)
    verify_inference_receipt(reasoner.artifact_store,
                              proposed.inference_receipt_ref, reasoner.contract)
    selected = planner_selection(proposed.answer, proposed.inference_receipt_ref)
    chosen = ClosedLoopExperimentPlanner().admit_model_selection(
        reviewed_diagnosis, selected,
        set(reg.validate(selected.evidence_refs, run_id=RUN, lineage_id=LINEAGE).valid_refs),
    )
    if chosen.intervention_kind not in ALLOWED_RESEARCH_DECISIONS:
        raise PinnedEvidenceError("scientific intervention outside declared campaign")
    decision = {
        "schema":"prometheus-authentic-scientific-decision.v1",
        "research_owner":"HEPHAESTUS_ACTUAL_FROZEN_ROLE_INFERENCE",
        "scientific_diagnosis":reviewed_diagnosis.to_dict(),
        "model_diagnosis_receipt_ref":diagnosis_observation.inference_receipt_ref,
        "model_planner_receipt_ref":proposed.inference_receipt_ref,
        "model_diagnosis_identity":{
            "model_id":diagnosis_observation.model_id,
            "revision":diagnosis_observation.revision,
            "adapter_sha256":diagnosis_observation.adapter_sha256,
            "output_sha256":diagnosis_observation.output_sha256,
        },
        "model_planner_identity":{
            "model_id":proposed.model_id,
            "revision":proposed.revision,
            "adapter_sha256":proposed.adapter_sha256,
            "output_sha256":proposed.output_sha256,
        },
        "model_planner_selection":selected.to_dict(),
        "independently_admitted_intervention":chosen.to_dict(),
        "training_executed":False,
        "original_parent_evaluated_on_novel_data":False,
        "independent_certification":False,
        "source_parent_report_sha256":REF_PARENT,
        "source_historical_d_report_sha256":REF_D,
        "next_step_requires_separate_admitted_executor":True,
    }
    (out/"scientific-decision.json").write_bytes(canonical_bytes(decision))
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--d-report", type=Path, required=True)
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter-cache", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/models/hephaestus_role_model_stack_v1.json"))
    parser.add_argument("--contract", type=Path, default=Path("configs/models/hephaestus_role_mastery_runtime_v1.json"))
    args = parser.parse_args(argv)
    parent = _verify(args.parent_report, PARENT_SHA)
    d = _verify(args.d_report, D_SHA)
    # All identity checks and deterministic proposals precede any GPU/remote access.
    prepare(parent, d)
    import boto3
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    generator = PinnedTransformersRoleGenerator(args.hf_cache,
        device="cuda:0", allow_remote_pinned_download=True)
    reasoner = CertifiedRoleReasoner(
        stack=load_certified_role_model_stack(args.registry),
        contract=RoleMasteryContract.load(args.contract),
        adapters=VerifiedRoleAdapters(
            boto3.client("s3", endpoint_url=args.s3_endpoint),
            args.s3_bucket, args.adapter_cache),
        generator=generator,
        artifact_store=FileSystemArtifactStore(args.out/"inference-receipts"),
    )
    try:
        result = decide(parent, d, reasoner=reasoner, out=args.out)
    finally:
        generator.close()
    print(json.dumps({
        "science_owner":result["research_owner"],
        "selection":result["independently_admitted_intervention"]["intervention_kind"],
        "diagnosis_receipt":result["model_diagnosis_receipt_ref"],
        "planner_receipt":result["model_planner_receipt_ref"],
        "training_executed":False,
        "final_assessment":False,
    },sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
