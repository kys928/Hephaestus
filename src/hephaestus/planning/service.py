"""Closed-loop planner that proposes, but never executes, experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from hephaestus.policy.experiment_policy import ExperimentPolicy, InterventionAssessment
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosticHypothesis
from hephaestus.schemas.discovery_contract import (
    DatasetSearchRequest,
    DatasetSelectionDecision,
    ModelSearchRequest,
    ModelSelectionDecision,
)
from hephaestus.schemas.experiment_contract import ExperimentProposal, InterventionProposal


class ExperimentPlanningError(ValueError):
    """Raised when shared contracts cannot represent a safe experiment."""


@runtime_checkable
class PlanningMemoryQuery(Protocol):
    def dead_ends_for_lineage(self, lineage_id: str, limit: int = 20) -> list[dict[str, object]]: ...

    def similar_failure_patterns(
        self,
        lineage_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]: ...

    def intervention_history(self, lineage_id: str, limit: int = 20) -> list[dict[str, object]]: ...


@dataclass(slots=True)
class _Candidate:
    proposal: InterventionProposal
    assessment: InterventionAssessment
    hypothesis_id: str


_DOMAIN_INTERVENTIONS: dict[str, tuple[str, ...]] = {
    "evaluation_integrity": ("repair_evaluation", "collect_more_evidence"),
    "launch_or_reproducibility": ("collect_more_evidence",),
    "data_quality": ("repair_data", "collect_more_evidence"),
    "data_coverage": ("replace_or_mix_dataset", "collect_more_evidence"),
    "data_format_or_wrapper": ("change_preprocessing", "collect_more_evidence"),
    "tokenizer": ("change_tokenizer", "collect_more_evidence"),
    "architecture": ("change_model", "collect_more_evidence"),
    "optimizer_or_scheduler": ("change_training_recipe", "collect_more_evidence"),
    "numerical_instability": ("change_training_recipe", "rollback"),
    "undertraining": ("resume_training", "change_training_recipe"),
    "overfitting": ("change_training_recipe", "rollback"),
    "decoding": ("collect_more_evidence",),
    "runtime_or_system": ("collect_more_evidence", "rollback"),
    "checkpoint_integrity": ("rollback", "branch"),
    "model_family_limitation": ("change_model", "branch", "restart"),
    "inconclusive": ("collect_more_evidence",),
}

_STAGE_PREFERENCE: dict[str, dict[str, float]] = {
    "tokenizer_validation": {"change_tokenizer": 1.0, "collect_more_evidence": 0.9, "repair_evaluation": 0.9},
    "smoke_test": {"collect_more_evidence": 1.0, "repair_data": 0.9, "change_training_recipe": 0.8},
    "early_pretraining": {"repair_data": 0.9, "replace_or_mix_dataset": 0.9, "change_training_recipe": 0.9},
    "scale_up_pretraining": {"resume_training": 0.9, "change_training_recipe": 0.9, "rollback": 0.8},
    "stabilization": {"collect_more_evidence": 1.0, "rollback": 0.9, "branch": 0.8},
    "continuation_repair": {"change_training_recipe": 0.9, "rollback": 0.9, "collect_more_evidence": 0.9},
    "ranking_repair": {"repair_evaluation": 0.9, "change_training_recipe": 0.9, "rollback": 0.8},
    "wrapper_specialization": {"change_preprocessing": 1.0, "collect_more_evidence": 0.9},
}

_ROLLBACK_PLANS = {
    "collect_more_evidence": "Discard diagnostic outputs and retain the unchanged baseline.",
    "repair_evaluation": "Restore the frozen evaluation reference and invalidate results from the revised evaluation protocol.",
    "repair_data": "Restore the prior immutable data manifest and preprocessing artifacts.",
    "replace_or_mix_dataset": "Restore the prior approved dataset selection and immutable mixture manifest.",
    "change_preprocessing": "Restore the prior preprocessing profile and regenerate data from the immutable source manifest.",
    "change_tokenizer": "Return to the baseline tokenizer and its compatible checkpoint lineage.",
    "change_training_recipe": "Stop the candidate run and continue from the unchanged baseline checkpoint and recipe.",
    "resume_training": "Interrupt the resumed run and restore the original checkpoint reference.",
    "change_model": "Reject the candidate model branch and retain the baseline model lineage.",
    "rollback": "Retain the pre-rollback lineage record and require approval before any forward transition.",
    "branch": "Archive or reject the child branch without mutating the parent lineage.",
    "restart": "Quarantine the restarted lineage and retain the prior lineage evidence for audit.",
    "stop": "No rollback is required because the planner does not execute the stop recommendation.",
}


def _stable_id(prefix: str, *values: object) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _unique_strings(values: object) -> list[str]:
    if values is None:
        items: Sequence[object] = ()
    elif isinstance(values, (str, bytes)):
        items = (values,)
    elif isinstance(values, Sequence):
        items = values
    else:
        items = (values,)
    return sorted({str(value) for value in items if str(value)})


@dataclass(slots=True)
class ClosedLoopExperimentPlanner:
    """Evidence-backed implementation of the frozen planning service protocol."""

    policy: ExperimentPolicy = field(default_factory=ExperimentPolicy)
    memory_query: PlanningMemoryQuery | None = None
    memory_limit: int = 20

    def propose_interventions(self, diagnosis: DiagnosisReport) -> Sequence[InterventionProposal]:
        memories = self._relevant_memories(diagnosis)
        hypotheses = self._ordered_hypotheses(diagnosis)
        candidates: list[_Candidate] = []

        for hypothesis in hypotheses:
            kinds = self._intervention_kinds(hypothesis)
            for kind in kinds:
                candidates.append(self._candidate(diagnosis, hypothesis, kind, memories))

        if diagnosis.missing_evidence or not hypotheses:
            synthetic = DiagnosticHypothesis(
                hypothesis_id="missing-evidence",
                failure_domain="inconclusive",
                summary="Resolve material missing evidence before selecting a training intervention.",
                required_tests=list(diagnosis.missing_evidence),
                recommended_intervention_kinds=["collect_more_evidence"],
                confidence=max(0.1, diagnosis.confidence),
            )
            candidates.append(self._candidate(diagnosis, synthetic, "collect_more_evidence", memories))

        deduped: dict[tuple[str, str, str], _Candidate] = {}
        for candidate in candidates:
            key = (
                candidate.proposal.intervention_kind,
                candidate.proposal.primary_variable,
                candidate.hypothesis_id,
            )
            previous = deduped.get(key)
            if previous is None or candidate.assessment.score > previous.assessment.score:
                deduped[key] = candidate
        candidates = list(deduped.values())

        rejected = {
            candidate.proposal.intervention_id: str(candidate.assessment.rejection_reason)
            for candidate in candidates
            if not candidate.assessment.accepted
        }
        accepted = [candidate for candidate in candidates if candidate.assessment.accepted]
        accepted.sort(
            key=lambda item: (
                -item.assessment.score,
                item.proposal.intervention_kind,
                item.proposal.primary_variable,
                item.proposal.intervention_id,
            )
        )

        if not accepted:
            fallback_hypothesis = hypotheses[0] if hypotheses else DiagnosticHypothesis(
                hypothesis_id="inconclusive",
                failure_domain="inconclusive",
                summary="No acceptable intervention is supported by current evidence.",
                confidence=diagnosis.confidence,
            )
            fallback = self._candidate(
                diagnosis,
                fallback_hypothesis,
                "collect_more_evidence",
                memories,
                ignore_dead_end=True,
            )
            accepted = [fallback]

        considered = [
            {
                "intervention_id": item.proposal.intervention_id,
                "kind": item.proposal.intervention_kind,
                "primary_variable": item.proposal.primary_variable,
                "score": item.assessment.score,
                "accepted": item.assessment.accepted,
                "rejection_reason": item.assessment.rejection_reason,
            }
            for item in sorted(candidates, key=lambda item: item.proposal.intervention_id)
        ]
        output: list[InterventionProposal] = []
        for rank, candidate in enumerate(accepted, start=1):
            candidate.proposal.alternatives_rejected = dict(sorted(rejected.items()))
            candidate.proposal.metadata.update(
                {
                    "rank": rank,
                    "candidate_count": len(candidates),
                    "alternatives_considered": considered,
                    "estimate_semantics": "heuristic_ordering_not_calibrated_probability",
                }
            )
            output.append(candidate.proposal)
        return output

    def create_discovery_requests(
        self,
        diagnosis: DiagnosisReport,
        intervention: InterventionProposal,
    ) -> tuple[DatasetSearchRequest | None, ModelSearchRequest | None]:
        validation = self.policy.validate_intervention(intervention)
        if validation:
            raise ExperimentPlanningError(";".join(validation))
        evidence_refs = self._evidence_refs(diagnosis)
        problem = intervention.hypothesis
        dataset_request: DatasetSearchRequest | None = None
        model_request: ModelSearchRequest | None = None
        metadata = dict(diagnosis.metadata)

        if intervention.intervention_kind == "replace_or_mix_dataset":
            dataset_request = DatasetSearchRequest(
                request_id=_stable_id("dataset-search", diagnosis.report_id, intervention.intervention_id),
                diagnosis_report_id=diagnosis.report_id,
                problem_statement=problem,
                capability_targets=_unique_strings(metadata.get("capability_targets", [problem])),
                required_languages=_unique_strings(metadata.get("required_languages", [])),
                required_domains=_unique_strings(metadata.get("required_domains", [])),
                required_formats=_unique_strings(metadata.get("required_formats", [])),
                tokenizer_ref=str(metadata.get("tokenizer_ref")) if metadata.get("tokenizer_ref") else None,
                model_constraints=dict(metadata.get("model_constraints", {})),
                size_constraints=dict(metadata.get("dataset_size_constraints", {})),
                license_allowlist=_unique_strings(metadata.get("dataset_license_allowlist", [])),
                license_denylist=_unique_strings(metadata.get("dataset_license_denylist", [])),
                provider_allowlist=_unique_strings(metadata.get("dataset_provider_allowlist", [])),
                evidence_refs=evidence_refs,
                metadata={"intervention_id": intervention.intervention_id},
            )
        if intervention.intervention_kind == "change_model":
            model_request = ModelSearchRequest(
                request_id=_stable_id("model-search", diagnosis.report_id, intervention.intervention_id),
                diagnosis_report_id=diagnosis.report_id,
                problem_statement=problem,
                task_requirements=_unique_strings(metadata.get("task_requirements", [problem])),
                architecture_constraints=dict(metadata.get("architecture_constraints", {})),
                tokenizer_constraints=dict(metadata.get("tokenizer_constraints", {})),
                runtime_constraints=dict(metadata.get("runtime_constraints", {})),
                budget_constraints=dict(metadata.get("budget", {})),
                license_allowlist=_unique_strings(metadata.get("model_license_allowlist", [])),
                provider_allowlist=_unique_strings(metadata.get("model_provider_allowlist", [])),
                evidence_refs=evidence_refs,
                metadata={"intervention_id": intervention.intervention_id},
            )
        return dataset_request, model_request

    def propose_experiment(
        self,
        diagnosis: DiagnosisReport,
        intervention: InterventionProposal,
        dataset_selection: DatasetSelectionDecision | None,
        model_selection: ModelSelectionDecision | None,
    ) -> ExperimentProposal:
        validation = self.policy.validate_intervention(intervention)
        if validation:
            raise ExperimentPlanningError(";".join(validation))
        if intervention.diagnosis_report_id != diagnosis.report_id:
            raise ExperimentPlanningError("intervention_diagnosis_mismatch")
        if intervention.intervention_kind == "stop":
            raise ExperimentPlanningError("stop_is_a_recommendation_not_an_experiment")

        baseline_ref = str(
            intervention.metadata.get("baseline_ref") or diagnosis.metadata.get("baseline_ref") or ""
        ) or None
        baseline_justification = str(
            intervention.metadata.get("baseline_justification")
            or diagnosis.metadata.get("baseline_justification")
            or ""
        )
        if baseline_ref is None and not baseline_justification:
            raise ExperimentPlanningError("baseline_required_or_explicit_justification")

        dataset_selection_id: str | None = None
        model_selection_id: str | None = None
        selection_evidence: list[str] = []
        approvals = set(str(item) for item in intervention.metadata.get("required_approvals", []))
        if intervention.intervention_kind == "replace_or_mix_dataset":
            if dataset_selection is None or dataset_selection.status != "selected":
                raise ExperimentPlanningError("selected_dataset_decision_required")
            if not dataset_selection.selected_candidate_ids:
                raise ExperimentPlanningError("dataset_selection_has_no_selected_candidates")
            expected_request, _ = self.create_discovery_requests(diagnosis, intervention)
            if expected_request is None or dataset_selection.request_id != expected_request.request_id:
                raise ExperimentPlanningError("dataset_selection_request_mismatch")
            if any(issue.blocking for issue in dataset_selection.issues):
                raise ExperimentPlanningError("dataset_selection_contains_blocking_issue")
            dataset_selection_id = dataset_selection.decision_id
            approvals.update(dataset_selection.required_approvals)
            selection_evidence.extend(dataset_selection.evidence_refs)
        elif dataset_selection is not None:
            raise ExperimentPlanningError("dataset_selection_not_required_for_intervention")

        if intervention.intervention_kind == "change_model":
            if model_selection is None or model_selection.status != "selected":
                raise ExperimentPlanningError("selected_model_decision_required")
            if not model_selection.selected_candidate_id:
                raise ExperimentPlanningError("model_selection_has_no_selected_candidate")
            _, expected_request = self.create_discovery_requests(diagnosis, intervention)
            if expected_request is None or model_selection.request_id != expected_request.request_id:
                raise ExperimentPlanningError("model_selection_request_mismatch")
            if any(issue.blocking for issue in model_selection.issues):
                raise ExperimentPlanningError("model_selection_contains_blocking_issue")
            model_selection_id = model_selection.decision_id
            approvals.update(model_selection.required_approvals)
            selection_evidence.extend(model_selection.evidence_refs)
        elif model_selection is not None:
            raise ExperimentPlanningError("model_selection_not_required_for_intervention")

        diagnostic_only = intervention.intervention_kind in {"collect_more_evidence", "repair_evaluation"}
        training_constraints = dict(diagnosis.metadata.get("training_constraints", {}))
        training_constraints.update(dict(intervention.metadata.get("training_constraints", {})))
        training_constraints["training_required"] = not diagnostic_only
        training_constraints["one_primary_variable"] = intervention.primary_variable

        required_evidence = _unique_strings(
            [
                *self._evidence_refs(diagnosis),
                *selection_evidence,
                *intervention.required_inputs,
                "baseline_comparison",
                "deterministic_scorecard",
                "frozen_eval_pack",
                "diagnostic_result" if diagnostic_only else "training_run_evidence",
            ]
        )
        budget = dict(diagnosis.metadata.get("budget", {}))
        budget.setdefault("estimated_cost", intervention.expected_cost)
        budget.setdefault("enforcement", "downstream_readiness_and_training_services")

        experiment_id = _stable_id(
            "experiment",
            diagnosis.report_id,
            intervention.intervention_id,
            dataset_selection_id,
            model_selection_id,
            baseline_ref,
        )
        return ExperimentProposal(
            experiment_id=experiment_id,
            run_id=_stable_id("planned-run", experiment_id),
            lineage_id=diagnosis.lineage_id,
            stage_name=diagnosis.stage_name,
            diagnosis_report_id=diagnosis.report_id,
            intervention_id=intervention.intervention_id,
            primary_variable=intervention.primary_variable,
            baseline_ref=baseline_ref,
            dataset_selection_id=dataset_selection_id,
            model_selection_id=model_selection_id,
            controlled_variables=dict(intervention.controlled_variables),
            training_constraints=training_constraints,
            budget=budget,
            success_criteria=dict(intervention.success_criteria),
            failure_criteria=dict(intervention.failure_criteria),
            required_evidence=required_evidence,
            required_approvals=sorted(approvals),
            rollback_plan=intervention.rollback_plan,
            status="pending",
            metadata={
                "hypothesis": intervention.hypothesis,
                "baseline_justification": baseline_justification,
                "alternatives_rejected": dict(intervention.alternatives_rejected),
                "planner_rank": intervention.metadata.get("rank"),
                "planner_score": intervention.metadata.get("ranking_score"),
                "planner_does_not_execute": True,
                "approval_pending": bool(approvals),
            },
        )

    def _candidate(
        self,
        diagnosis: DiagnosisReport,
        hypothesis: DiagnosticHypothesis,
        kind: str,
        memories: list[dict[str, object]],
        *,
        ignore_dead_end: bool = False,
    ) -> _Candidate:
        primary = self._primary_variable(kind, hypothesis)
        controlled = self._controlled_variables(diagnosis, primary)
        evidence_refs = _unique_strings(
            [*hypothesis.supporting_evidence_refs, *self._evidence_refs(diagnosis)]
        )
        matching_dead_ends = [
            memory for memory in memories if self._memory_matches(memory, kind, primary, hypothesis)
            and str(memory.get("memory_type")) == "known_dead_end"
        ]
        prior_attempts = sum(
            1 for memory in memories if self._memory_matches(memory, kind, primary, hypothesis)
        )
        memory_evidence = {
            str(ref) for memory in matching_dead_ends for ref in memory.get("evidence_refs", [])
        }
        explicit_new = {str(item) for item in diagnosis.metadata.get("new_evidence_refs", [])}
        has_new_evidence = bool((set(evidence_refs) | explicit_new) - memory_evidence) and bool(memory_evidence)
        known_dead_end = bool(matching_dead_ends) and not ignore_dead_end
        completeness = self._evidence_completeness(diagnosis)
        baseline_quality = _clamp(
            diagnosis.metadata.get("baseline_quality", 0.7 if diagnosis.metadata.get("baseline_ref") else 0.0)
        )
        information_gain = 0.95 if diagnosis.missing_evidence and kind == "collect_more_evidence" else 0.65
        if kind in {"repair_evaluation", "collect_more_evidence"}:
            information_gain = max(information_gain, 0.80)
        stage_appropriateness = _STAGE_PREFERENCE.get(diagnosis.stage_name, {}).get(kind, 0.65)
        trust_level = str(diagnosis.metadata.get("lineage_trust_level", "unknown"))
        approvals = self.policy.required_approvals(kind, diagnosis.stage_name, trust_level)
        assessment = self.policy.assess(
            intervention_kind=kind,
            hypothesis_confidence=_clamp(hypothesis.confidence),
            information_gain=information_gain,
            evidence_completeness=completeness,
            baseline_quality=baseline_quality,
            stage_appropriateness=stage_appropriateness,
            prior_attempts=prior_attempts,
            known_dead_end=known_dead_end,
            has_new_evidence=has_new_evidence,
            missing_evidence_count=len(diagnosis.missing_evidence),
            required_approvals=approvals,
        )
        success, failure = self._criteria(kind, hypothesis)
        intervention_id = _stable_id(
            "intervention",
            diagnosis.report_id,
            hypothesis.hypothesis_id,
            kind,
            primary,
        )
        proposal = InterventionProposal(
            intervention_id=intervention_id,
            diagnosis_report_id=diagnosis.report_id,
            intervention_kind=kind,
            hypothesis=hypothesis.summary,
            primary_variable=primary,
            controlled_variables=controlled,
            expected_effect=f"Test whether changing only {primary} addresses: {hypothesis.summary}",
            expected_cost=assessment.expected_cost,
            required_inputs=_unique_strings([*evidence_refs, *diagnosis.missing_evidence]),
            risks=self._risks(kind, hypothesis, known_dead_end),
            success_criteria=success,
            failure_criteria=failure,
            rollback_plan=_ROLLBACK_PLANS[kind],
            confidence=assessment.estimate_confidence,
            metadata={
                "hypothesis_id": hypothesis.hypothesis_id,
                "failure_domain": hypothesis.failure_domain,
                "ranking_score": assessment.score,
                "ranking_score_components": assessment.score_components,
                "estimate_confidence": assessment.estimate_confidence,
                "required_approvals": list(assessment.required_approvals),
                "prior_attempt_count": prior_attempts,
                "known_dead_end_matches": _unique_strings(
                    [memory.get("memory_id", "") for memory in matching_dead_ends]
                ),
                "new_evidence_overrides_dead_end": has_new_evidence,
                "changed_variables": [primary],
                "baseline_ref": diagnosis.metadata.get("baseline_ref"),
            },
        )
        return _Candidate(proposal, assessment, hypothesis.hypothesis_id)

    def _relevant_memories(self, diagnosis: DiagnosisReport) -> list[dict[str, object]]:
        if self.memory_query is None:
            return []
        rows = [
            *self.memory_query.dead_ends_for_lineage(diagnosis.lineage_id, limit=self.memory_limit),
            *self.memory_query.similar_failure_patterns(
                lineage_id=diagnosis.lineage_id,
                limit=self.memory_limit,
            ),
            *self.memory_query.intervention_history(diagnosis.lineage_id, limit=self.memory_limit),
        ]
        deduped: dict[str, dict[str, object]] = {}
        for index, row in enumerate(rows):
            key = str(row.get("memory_id") or f"anonymous-{index}")
            deduped[key] = row
        return [deduped[key] for key in sorted(deduped)]

    @staticmethod
    def _ordered_hypotheses(diagnosis: DiagnosisReport) -> list[DiagnosticHypothesis]:
        return sorted(
            diagnosis.hypotheses,
            key=lambda item: (
                0 if item.hypothesis_id == diagnosis.leading_hypothesis_id else 1,
                -item.confidence,
                item.hypothesis_id,
            ),
        )

    @staticmethod
    def _intervention_kinds(hypothesis: DiagnosticHypothesis) -> tuple[str, ...]:
        values = hypothesis.recommended_intervention_kinds or list(
            _DOMAIN_INTERVENTIONS.get(hypothesis.failure_domain, ("collect_more_evidence",))
        )
        return tuple(dict.fromkeys(str(item) for item in values))

    @staticmethod
    def _primary_variable(kind: str, hypothesis: DiagnosticHypothesis) -> str:
        summary = hypothesis.summary.lower()
        if kind == "collect_more_evidence":
            return "decoding_setting" if hypothesis.failure_domain == "decoding" else "diagnostic_measurement"
        if kind == "repair_evaluation":
            return "evaluation_protocol"
        if kind == "repair_data":
            return "data_quality_filter"
        if kind == "replace_or_mix_dataset":
            return "dataset_mixture"
        if kind == "change_preprocessing":
            return "preprocessing_policy"
        if kind == "change_tokenizer":
            return "tokenizer"
        if kind == "change_training_recipe":
            if "scheduler" in summary or "warmup" in summary or "decay" in summary:
                return "scheduler"
            if "batch" in summary or "accumulation" in summary:
                return "batch_construction"
            if hypothesis.failure_domain in {"undertraining", "overfitting"} or "duration" in summary:
                return "training_duration"
            return "learning_rate"
        if kind == "resume_training":
            return "checkpoint_resume_point"
        if kind == "change_model":
            return "model_candidate"
        if kind in {"rollback", "branch", "restart"}:
            return "checkpoint_resume_point"
        if kind == "stop":
            return "termination_decision"
        return "diagnostic_measurement"

    @staticmethod
    def _controlled_variables(diagnosis: DiagnosisReport, primary: str) -> dict[str, object]:
        metadata = diagnosis.metadata
        controls: dict[str, object] = {
            "all_non_primary_variables": "held_constant_unless_documented_unavoidable",
            "random_seed": metadata.get("random_seed", "held_constant"),
            "evaluation_reference": metadata.get("eval_pack_ref", "held_constant"),
            "baseline_decoding": metadata.get("decoding_ref", "held_constant"),
            "architecture": metadata.get("architecture_ref", "held_constant"),
            "tokenizer": metadata.get("tokenizer_ref", "held_constant"),
            "dataset_manifest": metadata.get("dataset_manifest_ref", "held_constant"),
            "training_recipe": metadata.get("training_recipe_ref", "held_constant"),
        }
        controls.pop(primary, None)
        return controls

    @staticmethod
    def _criteria(
        kind: str,
        hypothesis: DiagnosticHypothesis,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if kind in {"collect_more_evidence", "repair_evaluation"}:
            return (
                {
                    "required_tests_completed": list(hypothesis.required_tests) or ["diagnostic_comparison"],
                    "hypothesis_resolution": "supported_or_rejected_with_persisted_evidence",
                    "deterministic_evidence_present": True,
                },
                {
                    "required_test_missing": True,
                    "result_remains_inconclusive": True,
                    "evidence_integrity_failure": True,
                },
            )
        return (
            {
                "stage_primary_metric_delta_min": 0.01,
                "deterministic_regression_count_max": 0,
                "baseline_comparison_required": True,
            },
            {
                "stage_primary_metric_delta_max": 0.0,
                "deterministic_regression_count_min": 1,
                "budget_exhausted_without_evaluable_result": True,
            },
        )

    @staticmethod
    def _risks(
        kind: str,
        hypothesis: DiagnosticHypothesis,
        known_dead_end: bool,
    ) -> list[str]:
        risks = ["heuristic_expected_value_estimate", "baseline_or_eval_mismatch"]
        if kind in {"replace_or_mix_dataset", "repair_data", "change_preprocessing"}:
            risks.extend(["data_lineage_corruption", "contamination_or_provenance_risk"])
        if kind in {"change_tokenizer", "change_model"}:
            risks.append("checkpoint_or_contract_incompatibility")
        if kind in {"restart", "change_model", "change_tokenizer"}:
            risks.append("high_reversal_cost")
        if hypothesis.contradicting_evidence_refs:
            risks.append("contradicting_evidence_present")
        if known_dead_end:
            risks.append("known_dead_end_repeat")
        return sorted(set(risks))

    @staticmethod
    def _evidence_refs(diagnosis: DiagnosisReport) -> list[str]:
        return _unique_strings(
            [
                *(observation.source_ref for observation in diagnosis.observations),
                *(ref for hypothesis in diagnosis.hypotheses for ref in hypothesis.supporting_evidence_refs),
                *(ref for hypothesis in diagnosis.hypotheses for ref in hypothesis.contradicting_evidence_refs),
            ]
        )

    @staticmethod
    def _evidence_completeness(diagnosis: DiagnosisReport) -> float:
        present = len(diagnosis.observations) + sum(
            len(item.supporting_evidence_refs) for item in diagnosis.hypotheses
        )
        missing = len(diagnosis.missing_evidence)
        if present + missing == 0:
            return 0.0
        return round(present / (present + missing), 6)

    @staticmethod
    def _memory_matches(
        memory: dict[str, object],
        kind: str,
        primary: str,
        hypothesis: DiagnosticHypothesis,
    ) -> bool:
        metadata = dict(memory.get("metadata", {}))
        lineage_dead_end = metadata.get("lineage_status") in {"poisoned", "deprecated", "archived"}
        if lineage_dead_end:
            return kind not in {"branch", "restart", "stop", "collect_more_evidence"}
        tags = {str(item).lower() for item in memory.get("tags", [])}
        summary = str(memory.get("summary", "")).lower()
        direct = {
            str(metadata.get("intervention_kind", "")).lower(),
            str(metadata.get("primary_variable", "")).lower(),
        }
        targets = {kind.lower(), primary.lower(), hypothesis.hypothesis_id.lower()}
        return bool(targets.intersection(tags | direct)) or kind.lower() in summary or primary.lower() in summary
