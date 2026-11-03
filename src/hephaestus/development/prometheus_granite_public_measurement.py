"""Authentic frozen Granite Evaluator: public six-state diagnostic AFTER an admitted model decision.

This narrow executor cannot run from a Sol-authored intervention string. It
verifies the Diagnostician/Planner's actual CUDA role-inference receipts and
deterministic Planner admission, then measures the unchanged frozen Evaluator
on the already authored, public 120-case development probe.

This is NOT protected confirmation, model repair, or production promotion.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.development.granite_sixstate_public_probe import (
    build_proposed_pack, canonical_hash, validate_proposed_pack,
)
from hephaestus.development.prometheus_scientific_decision import (
    _require_actual_gpu_receipt, REF_PARENT, REF_D,
    RUN, LINEAGE, STAGE, PARENT_ADAPTER, PARENT_REVISION,
)
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError, _parse_json
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.llm.role_contract import RoleEvidence, RoleTask
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner
from hephaestus.storage.verified import canonical_bytes

SCHEMA = "prometheus-frozen-granite-public-development-measurement.v1"
TOOL_OWNER = "HEPHAESTUS_DETERMINISTIC_EXECUTION_POLICY"
MEASUREMENT_ID = "public-sixstate-paired-counterfactual-v1"
MAX_PUBLIC_CASES = 120


def admit_decision(reasoner: CertifiedRoleReasoner, decision_root: Path) -> dict:
    source = decision_root / "scientific-decision.json"
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 150_000:
        raise PinnedEvidenceError("no bounded model-originated decision artifact exists")
    value = _parse_json(source.read_text())
    if not isinstance(value, dict) or value.get("schema") != "prometheus-authentic-scientific-decision.v1":
        raise PinnedEvidenceError("scientific decision artifact is invalid")
    if (value.get("research_owner") != "HEPHAESTUS_ACTUAL_FROZEN_ROLE_INFERENCE"
            or value.get("source_parent_report_sha256") != REF_PARENT
            or value.get("source_historical_d_report_sha256") != REF_D
            or value.get("training_executed") is not False
            or value.get("independent_certification") is not False):
        raise PinnedEvidenceError("scientific decision source or ownership invalid")
    admitted = value.get("independently_admitted_intervention")
    if not isinstance(admitted, dict) or admitted.get("intervention_kind") != "collect_more_evidence":
        raise PermissionError("Planner did not select an admissible evidence collection action")
    if admitted.get("primary_variable") != "diagnostic_measurement":
        raise PermissionError("Planner did not select the diagnostic measurement variable")
    for role,field in (
        ("diagnosis","model_diagnosis_receipt_ref"), ("planner","model_planner_receipt_ref")
    ):
        ref = value.get(field)
        if not isinstance(ref,str) or not ref.startswith("sha256:"):
            raise PinnedEvidenceError("missing original model inference receipt")
        _require_actual_gpu_receipt(reasoner, ref, role)
    provenance = admitted.get("metadata", {}).get("model_inference_receipt_ref")
    if provenance != value["model_planner_receipt_ref"]:
        raise PinnedEvidenceError("admitted intervention not bound to actual Planner receipt")
    # This is a deterministic tool mapping, NOT a free-choice research-tool
    # recommendation from the frozen seven-key Planner vocabulary.
    return value


def run_public_measurement(reasoner: CertifiedRoleReasoner, *,
                           decision_root: Path, output: Path) -> dict:
    decision = admit_decision(reasoner, decision_root)
    pack = build_proposed_pack()
    validate_proposed_pack(pack)
    if pack["case_count"] != MAX_PUBLIC_CASES or pack["research_execution_authorized"] is not False:
        raise PinnedEvidenceError("unrecognized or prematurely authorized public probe")
    spec = reasoner.stack.resolve("evaluator")
    if (spec.model_id != "ibm-granite/granite-3.3-8b-instruct"
            or spec.revision != PARENT_REVISION
            or spec.adapter.sha256 != PARENT_ADAPTER):
        raise PermissionError("nonoriginal Evaluator weights are forbidden")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("public diagnosis measurement cannot overwrite earlier results")
    output.mkdir(parents=True, exist_ok=True)
    precommit = {
        "schema":"prometheus-public-development-execution-precommit.v1",
        "diagnostic_id":MEASUREMENT_ID,
        "public_pack_sha256":"sha256:" + canonical_hash(pack),
        "case_count":MAX_PUBLIC_CASES,
        "selected_scientific_action":decision["independently_admitted_intervention"]["intervention_kind"],
        "decision_receipt_ref":decision["model_planner_receipt_ref"],
        "specific_tool_selection_owner":TOOL_OWNER,
        "original_certified_evaluator_adapter_sha256":PARENT_ADAPTER,
        "task_ontology":pack["ontology"],
        "task_partition":"public_development_proposal",
        "final_holdout_accessed":False,
        "independently_approved_oracle":False,
        "no_training_or_promotion":True,
    }
    (output/"precommit.json").write_bytes(canonical_bytes(precommit))
    per_state = {state:Counter() for state in sorted({
        r["reference"]["decision"] for r in pack["cases"]
    })}
    pair_results:dict[str,list[tuple[str,str,bool]]]={}
    cases = []
    schema_ok = 0
    eos_count = 0
    for row in pack["cases"]:
        factset = tuple(RoleEvidence(x["ref"],x["fact"]) for x in row["evidence"])
        registry = EvidenceRegistry({
            x.ref: EvidenceRecord(x.ref,"public_proposed_development_case",
                run_id=RUN,lineage_id=LINEAGE,state="verified")
            for x in factset
        })
        task = RoleTask("evaluator",row["situation"],factset,RUN,LINEAGE,STAGE)
        context = DeterministicRoleContext(RUN,LINEAGE,STAGE,
            approval_status="none",stage_allowed_actions=("observe_state","request_recheck"))
        try:
            obs = reasoner.propose(task,registry,context=context,max_new_tokens=384)
        except ValueError as exc:
            rejected = getattr(exc,"rejected_inference_receipt_ref",None)
            if not isinstance(rejected,str):
                raise
            receipt = verify_inference_receipt(reasoner.artifact_store,rejected,reasoner.contract)
            # Rejected raw input is stored privately. For a scientifically
            # meaningful malformed-response count, demand genuine CUDA traces.
            ex = receipt.get("execution",{})
            if (receipt.get("role")!="evaluator"
                or receipt.get("generator")!="PinnedTransformersRoleGenerator"
                or not isinstance(ex,dict)
                or ex.get("device")!="cuda:0"
                or not ex.get("generated_token_ids")):
                raise PermissionError("malformed output lacks genuine CUDA provenance") from exc
            response="<invalid>"
            model_ref=rejected
            eos_count += int(receipt.get("finish_reason")=="eos")
        else:
            receipt = _require_actual_gpu_receipt(reasoner,obs.inference_receipt_ref,"evaluator")
            response=obs.answer.decision
            model_ref=obs.inference_receipt_ref
            schema_ok += 1
            eos_count += int(receipt.get("finish_reason")=="eos")
        gold=row["reference"]["decision"]
        correct=response==gold
        per_state[gold][response]+=1
        pair_results.setdefault(row["pair_id"],[]).append((row["pair_side"],response,correct))
        # Never put raw model generations or prompts in the public aggregate.
        cases.append({"case_id":row["case_id"],
             "receipt_ref":model_ref,"expected_decision":gold,
             "observed_decision":response,"correct":correct})
    groups={r["pair_id"]:r["pair_kind"] for r in pack["cases"]}
    sensitivity=0
    invariance=0
    for pair_id, result in pair_results.items():
        if len(result)!=2 or {x[0] for x in result}!={"A","B"}:
            raise PinnedEvidenceError("one or more public diagnostic pairs is incomplete")
        if groups[pair_id]=="causal_single_factor":
            sensitivity += int(all(x[2] for x in result) and result[0][1]!=result[1][1])
        else:
            invariance += int(all(x[2] for x in result) and result[0][1]==result[1][1])
    correct=sum(x["correct"] for x in cases)
    output_doc={
        "schema":SCHEMA,
        "original_parent_model_id":spec.model_id,
        "original_parent_revision":spec.revision,
        "original_parent_adapter_sha256":spec.adapter.sha256,
        "scientific_action_owner":"HEPHAESTUS_ACTUAL_FROZEN_PLANNER",
        "specific_measurement_tool_owner":TOOL_OWNER,
        "public_pack_sha256":precommit["public_pack_sha256"],
        "n_cases":len(cases),"exact_correct":correct,"exact_accuracy":correct/len(cases),
        "valid_schema_count":schema_ok,"eos_count":eos_count,
        "causal_pair_both_correct":sensitivity,"causal_pair_total":48,
        "nuisance_pair_both_correct":invariance,"nuisance_pair_total":12,
        "per_state":{label:{"total":sum(c.values()),"correct":c[label],
                 "confusion":dict(sorted(c.items()))} for label,c in per_state.items()},
        "case_receipts":cases,
        "independent_final_holdout":False,
        "independently_accepted_public_oracle":False,
        "original_certification_score_unchanged":True,
        "training_steps":0,"model_promotion":False,
        "scientific_repair_demonstrated":False,
        "requires_followup_scientific_diagnosis":True,
    }
    (output/"public-measurement.json").write_bytes(canonical_bytes(output_doc))
    return output_doc
