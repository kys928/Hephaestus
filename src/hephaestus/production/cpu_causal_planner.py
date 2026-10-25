"""Research-only Planner bridge: actual CPU causal LM ranks REAL admissible plans.

Deterministic scientific policy determines the candidate set, not the model.
The pinned 135M causal language model ranks the safe policy-compatible
alternatives by actual conditional log likelihood. No proposal is an approval.
Every later Planner substep re-opens original persisted model and diagnosis
receipts and re-admits the same scientific intervention before use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
)
from hephaestus.llm.cpu_causal_research import CANDIDATES, SCOPE
from hephaestus.llm.role_contract import RoleAnswer
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.policy.model_proposals import PLANNER_DECISIONS, planner_selection
from hephaestus.schemas.model_proposal import ModelInterventionSelection
from hephaestus.storage.verified import canonical_bytes, read_verified_json


@dataclass(slots=True)
class CpuResearchCausalPlanner:
    handoff: object
    scorer: object
    evidence: object
    planner: ClosedLoopExperimentPlanner

    def _scientifically_admitted_options(self, diagnosis):
        domain = self.planner.propose_interventions(diagnosis)
        admissible = set()
        for decision, action, variable in CANDIDATES["planner"]:
            kind, policy_action, variable_map = PLANNER_DECISIONS[decision]
            if action != policy_action:
                continue
            real_variable = variable_map.get(variable)
            if any(row.intervention_kind == kind and
                   row.primary_variable == real_variable for row in domain):
                admissible.add((decision, action, variable))
        return tuple(option for option in CANDIDATES["planner"] if option in admissible)

    def _chosen(self, req):
        source = req.prior_outputs.get("planner.intervention_ranking", {})
        ref = source.get("selected_intervention_ref")
        if not isinstance(ref, str) or not ref.startswith("sha256:"):
            raise ValueError("no immutable selected intervention")
        artifact = read_verified_json(self.scorer.artifact_store, ref)
        select = ModelInterventionSelection.from_dict(artifact["selection"])
        actual = read_verified_json(self.scorer.artifact_store,
                                    select.inference_receipt_ref)
        if (actual.get("schema") != SCOPE or
                actual.get("role") != "planner" or
                actual.get("run_id") != req.run_id or
                actual.get("lineage_id") != req.lineage_id or
                actual.get("stage_name") != req.stage_name or
                not actual.get("advisory_only") or
                actual.get("action_executed")):
            raise ValueError("CPU causal planner source scope mismatch")
        choice = actual["selected_candidate"]
        if (choice["decision"] != select.model_decision["decision"]
                or choice["action"] != select.model_decision["action"]
                or choice["primary_variable"] != select.model_decision["primary_variable"]):
            raise ValueError("scientific intervention changed after LM scoring")
        verified = self.evidence.verify_refs(req, list(select.evidence_refs))
        report = self.evidence.diagnosis_for(req)
        chosen = self.planner.admit_model_selection(report, select, verified)
        if chosen.to_dict() != artifact["intervention"]:
            raise ValueError("reviewed intervention differs from persisted scientific proposal")
        return report, chosen

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep == "diagnosis_handoff":
            return self.handoff.execute(req)
        if req.substep == "intervention_ranking":
            report = self.evidence.diagnosis_for(req)
            options = self._scientifically_admitted_options(report)
            if not options:
                return StagedOperationResult(
                    status="blocked", resumable=False,
                    blocking_issues=("real_planner_has_no_cpu_lm_admissible_scientific_intervention",))
            task, registry, context = self.evidence.bind(role="planner", request=req)
            raw = self.scorer.propose(
                task, registry, context, admissible_candidates=options)
            chosen = raw["selected_candidate"]
            raw_scores = [r["mean_conditional_log_likelihood"]
                          for r in raw["candidates"]]
            max_score = max(raw_scores)
            denominator = sum(math.exp(score - max_score) for score in raw_scores)
            probability = math.exp(
                chosen["mean_conditional_log_likelihood"] - max_score) / denominator
            # A relative LM preference IS NOT confidence in experiment quality.
            answer = RoleAnswer(
                decision=chosen["decision"], action=chosen["action"],
                primary_variable=chosen["primary_variable"],
                confidence=probability,
                evidence_refs=tuple(r.ref for r in task.evidence),
                uncertainties=("CPU causal proposal likelihood is not calibrated scientific confidence",),
                rationale="A pinned real pretrained CPU causal LM ranked policy-admitted interventions.",
            )
            selection = planner_selection(answer, raw["inference_receipt_ref"])
            verified = self.evidence.verify_refs(req, selection.evidence_refs)
            admitted = self.planner.admit_model_selection(report, selection, verified)
            packet = self.scorer.artifact_store.put_bytes(
                canonical_bytes({
                    "selection":selection.to_dict(),
                    "intervention":admitted.to_dict(),
                }), media_type="application/json")
            return StagedOperationResult(
                records=(
                    StagedOutputRecord("intervention_proposal", admitted.to_dict()),
                    StagedOutputRecord("cpu_lm_scientific_proposal", {
                        "scorer_receipt_ref":raw["inference_receipt_ref"],
                        "intervention_id":admitted.intervention_id,
                        "advisory_only":True,
                        "scientific_certification":False,
                        "number_of_admissible_options":len(options),
                    }),
                ),
                output_refs=(packet.artifact_ref,raw["inference_receipt_ref"]),
                metadata={
                    "selected_intervention_ref":packet.artifact_ref,
                    "intervention_id":admitted.intervention_id,
                    "primary_variable":admitted.primary_variable,
                    "required_approvals":admitted.metadata["required_approvals"],
                    "cpu_causal_model_inference_ref":raw["inference_receipt_ref"],
                },
            )
        if req.substep in {
                "experiment_proposal", "discovery_request_preparation",
                "approval_requirement_discovery"}:
            report, chosen = self._chosen(req)
            if req.substep == "experiment_proposal":
                proposed = self.planner.propose_experiment(
                    report, chosen, None, None,
                    allow_unresolved_discovery=True)
                return StagedOperationResult(
                    records=(StagedOutputRecord("experiment_proposal",
                                                 proposed.to_dict()),),
                    output_refs=(proposed.experiment_id,),
                    metadata={
                        "experiment_id":proposed.experiment_id,
                        "intervention_id":chosen.intervention_id,
                        "primary_variable":chosen.primary_variable,
                        "unresolved_discovery":proposed.metadata["unresolved_discovery"],
                    },
                )
            if req.substep == "discovery_request_preparation":
                dataset, model = self.planner.create_discovery_requests(
                    report, chosen)
                return StagedOperationResult(
                    records=(StagedOutputRecord("search_request", {
                        "dataset_request":dataset.to_dict() if dataset else None,
                        "model_request":model.to_dict() if model else None,
                        "discovery_required":bool(dataset or model),
                    }),),
                    metadata={
                        "dataset_request_id":dataset.request_id if dataset else None,
                        "model_request_id":model.request_id if model else None,
                    },
                )
            approvals = list(chosen.metadata.get("required_approvals", []))
            return StagedOperationResult(
                records=(StagedOutputRecord("approval_requirements", {
                    "intervention_id":chosen.intervention_id,
                    "required_approvals":approvals,
                }),),
                metadata={"required_approvals":approvals},
            )
        return StagedOperationResult(
            status="blocked",
            blocking_issues=("unsupported_cpu_causal_planner_substep",),
            resumable=False)
