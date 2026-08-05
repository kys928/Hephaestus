from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hephaestus.interfaces import ExperimentPlanningService
from hephaestus.planning import ClosedLoopExperimentPlanner, ExperimentPlanningError
from hephaestus.policy.experiment_policy import ExperimentPolicy
from hephaestus.schemas.contract_common import CONTRACT_STATUSES, ContractIssue
from hephaestus.schemas.diagnosis_contract import (
    DiagnosisReport,
    DiagnosticHypothesis,
    EvidenceObservation,
)
from hephaestus.schemas.discovery_contract import DatasetSelectionDecision, ModelSelectionDecision
from hephaestus.schemas.experiment_contract import InterventionProposal


@dataclass
class FakeMemoryQuery:
    dead_ends: list[dict[str, object]] = field(default_factory=list)
    failures: list[dict[str, object]] = field(default_factory=list)
    interventions: list[dict[str, object]] = field(default_factory=list)

    def dead_ends_for_lineage(self, lineage_id: str, limit: int = 20) -> list[dict[str, object]]:
        return self.dead_ends[-limit:]

    def similar_failure_patterns(
        self,
        lineage_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        return self.failures[-limit:]

    def intervention_history(self, lineage_id: str, limit: int = 20) -> list[dict[str, object]]:
        return self.interventions[-limit:]


def _diagnosis(
    *,
    hypotheses: list[DiagnosticHypothesis] | None = None,
    missing_evidence: list[str] | None = None,
    confidence: float = 0.8,
) -> DiagnosisReport:
    hypotheses = hypotheses or [
        DiagnosticHypothesis(
            hypothesis_id="hyp-lr",
            failure_domain="optimizer_or_scheduler",
            summary="Learning rate is too high for the observed gradient scale.",
            supporting_evidence_refs=["evidence://gradients"],
            recommended_intervention_kinds=["change_training_recipe", "collect_more_evidence"],
            confidence=0.82,
        ),
        DiagnosticHypothesis(
            hypothesis_id="hyp-data",
            failure_domain="data_coverage",
            summary="The training mixture underrepresents the target capability.",
            supporting_evidence_refs=["evidence://coverage"],
            recommended_intervention_kinds=["replace_or_mix_dataset"],
            confidence=0.61,
        ),
    ]
    return DiagnosisReport(
        report_id="diag-1",
        request_id="diag-request-1",
        run_id="observed-run-1",
        lineage_id="lineage-1",
        stage_name="early_pretraining",
        status="completed",
        observations=[
            EvidenceObservation(
                observation_id="obs-1",
                evidence_kind="metric",
                source_ref="evidence://loss",
                summary="Loss became unstable.",
                confidence=0.9,
            )
        ],
        hypotheses=hypotheses,
        leading_hypothesis_id=hypotheses[0].hypothesis_id if hypotheses else None,
        missing_evidence=missing_evidence or [],
        confidence=confidence,
        metadata={
            "baseline_ref": "checkpoint://baseline",
            "baseline_quality": 0.8,
            "lineage_trust_level": "medium",
            "eval_pack_ref": "eval-pack://frozen/v1",
            "dataset_manifest_ref": "manifest://baseline",
            "training_recipe_ref": "recipe://baseline",
            "budget": {"max_compute_hours": 4.0},
        },
    )


def _proposal(planner: ClosedLoopExperimentPlanner, diagnosis: DiagnosisReport, kind: str) -> InterventionProposal:
    return next(item for item in planner.propose_interventions(diagnosis) if item.intervention_kind == kind)


def test_identical_inputs_produce_deterministic_ranking_and_ids() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()

    first = [item.to_dict() for item in planner.propose_interventions(diagnosis)]
    second = [item.to_dict() for item in planner.propose_interventions(diagnosis)]

    assert first == second
    assert [item["metadata"]["rank"] for item in first] == list(range(1, len(first) + 1))
    assert all(item["metadata"]["estimate_semantics"] == "heuristic_ordering_not_calibrated_probability" for item in first)
    assert {"compute", "data", "storage", "evaluation", "time"}.issubset(first[0]["expected_cost"])


def test_missing_evidence_yields_diagnostic_experiment_instead_of_training() -> None:
    diagnosis = _diagnosis(missing_evidence=["gradient_norm_series"], confidence=0.3)
    planner = ClosedLoopExperimentPlanner()

    interventions = list(planner.propose_interventions(diagnosis))
    assert interventions[0].intervention_kind == "collect_more_evidence"

    experiment = planner.propose_experiment(diagnosis, interventions[0], None, None)
    assert experiment.training_constraints["training_required"] is False
    assert experiment.primary_variable == "diagnostic_measurement"
    assert "diagnostic_result" in experiment.required_evidence


def test_known_dead_end_is_rejected_without_new_evidence_and_preserved() -> None:
    memory = {
        "memory_id": "mem-dead-lr",
        "memory_type": "known_dead_end",
        "summary": "change_training_recipe failed repeatedly",
        "tags": ["change_training_recipe", "learning_rate"],
        "evidence_refs": [],
        "metadata": {"intervention_kind": "change_training_recipe", "primary_variable": "learning_rate"},
    }
    planner = ClosedLoopExperimentPlanner(memory_query=FakeMemoryQuery(dead_ends=[memory]))

    interventions = list(planner.propose_interventions(_diagnosis(hypotheses=[_diagnosis().hypotheses[0]])))

    assert all(item.intervention_kind != "change_training_recipe" for item in interventions)
    assert any(
        reason == "known_dead_end_without_new_evidence"
        for item in interventions
        for reason in item.alternatives_rejected.values()
    )


def test_prior_attempts_reduce_score_but_do_not_fake_a_probability() -> None:
    prior = {
        "memory_id": "mem-prior-lr",
        "memory_type": "successful_intervention",
        "summary": "change_training_recipe attempt",
        "tags": ["change_training_recipe", "learning_rate"],
        "metadata": {"intervention_kind": "change_training_recipe"},
    }
    diagnosis = _diagnosis(hypotheses=[_diagnosis().hypotheses[0]])
    fresh = _proposal(ClosedLoopExperimentPlanner(), diagnosis, "change_training_recipe")
    repeated = _proposal(
        ClosedLoopExperimentPlanner(memory_query=FakeMemoryQuery(interventions=[prior])),
        diagnosis,
        "change_training_recipe",
    )

    assert repeated.metadata["ranking_score"] < fresh.metadata["ranking_score"]
    assert repeated.metadata["prior_attempt_count"] == 1


def test_one_primary_variable_rule_rejects_ambiguous_change() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    invalid = InterventionProposal(
        intervention_id="invalid",
        diagnosis_report_id=diagnosis.report_id,
        intervention_kind="change_training_recipe",
        hypothesis="Change multiple recipe fields.",
        primary_variable="learning_rate,scheduler",
        controlled_variables={"all_other_variables": "held_constant"},
        success_criteria={"metric_delta_min": 0.1},
        failure_criteria={"metric_delta_max": 0.0},
        rollback_plan="Restore baseline recipe.",
        metadata={"changed_variables": ["learning_rate", "scheduler"]},
    )

    with pytest.raises(ExperimentPlanningError, match="primary_variable"):
        planner.propose_experiment(diagnosis, invalid, None, None)


def test_discovery_requests_are_emitted_only_for_diagnosed_need() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    dataset_intervention = _proposal(planner, diagnosis, "replace_or_mix_dataset")
    recipe_intervention = _proposal(planner, diagnosis, "change_training_recipe")

    dataset_request, model_request = planner.create_discovery_requests(diagnosis, dataset_intervention)
    assert dataset_request is not None
    assert model_request is None

    dataset_request, model_request = planner.create_discovery_requests(diagnosis, recipe_intervention)
    assert dataset_request is None
    assert model_request is None

    model_diagnosis = _diagnosis(
        hypotheses=[
            DiagnosticHypothesis(
                hypothesis_id="hyp-model",
                failure_domain="model_family_limitation",
                summary="The current architecture cannot represent the target behavior.",
                recommended_intervention_kinds=["change_model"],
                supporting_evidence_refs=["evidence://capacity"],
                confidence=0.9,
            )
        ]
    )
    model_intervention = _proposal(planner, model_diagnosis, "change_model")
    dataset_request, model_request = planner.create_discovery_requests(model_diagnosis, model_intervention)
    assert dataset_request is None
    assert model_request is not None


def test_selected_dataset_becomes_complete_proposal_and_preserves_approvals() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    intervention = _proposal(planner, diagnosis, "replace_or_mix_dataset")
    dataset_request, _ = planner.create_discovery_requests(diagnosis, intervention)
    assert dataset_request is not None
    selection = DatasetSelectionDecision(
        decision_id="dataset-decision-1",
        request_id=dataset_request.request_id,
        status="selected",
        selected_candidate_ids=["dataset-candidate-1"],
        required_approvals=["license_review"],
    )

    experiment = planner.propose_experiment(diagnosis, intervention, selection, None)

    assert experiment.dataset_selection_id == selection.decision_id
    assert experiment.model_selection_id is None
    assert experiment.baseline_ref == "checkpoint://baseline"
    assert experiment.success_criteria and experiment.failure_criteria
    assert experiment.rollback_plan
    assert experiment.required_approvals == ["dataset_selection_approval", "license_review"]
    assert experiment.metadata["alternatives_rejected"] == intervention.alternatives_rejected


def test_selected_model_is_required_for_model_change() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis(
        hypotheses=[
            DiagnosticHypothesis(
                hypothesis_id="hyp-model",
                failure_domain="architecture",
                summary="Architecture limits the target capability.",
                recommended_intervention_kinds=["change_model"],
                supporting_evidence_refs=["evidence://architecture"],
                confidence=0.9,
            )
        ]
    )
    intervention = _proposal(planner, diagnosis, "change_model")
    _, model_request = planner.create_discovery_requests(diagnosis, intervention)
    assert model_request is not None

    with pytest.raises(ExperimentPlanningError, match="selected_model_decision_required"):
        planner.propose_experiment(diagnosis, intervention, None, None)

    decision = ModelSelectionDecision(
        decision_id="model-decision-1",
        request_id=model_request.request_id,
        status="selected",
        selected_candidate_id="model-candidate-1",
        required_approvals=["runtime_capacity_approval"],
    )
    experiment = planner.propose_experiment(diagnosis, intervention, None, decision)
    assert experiment.model_selection_id == decision.decision_id
    assert set(experiment.required_approvals) == {"model_selection_approval", "runtime_capacity_approval"}


def test_selection_must_match_the_exact_governed_request() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    intervention = _proposal(planner, diagnosis, "replace_or_mix_dataset")
    selection = DatasetSelectionDecision(
        decision_id="dataset-decision-wrong-request",
        request_id="dataset-search-for-another-intervention",
        status="selected",
        selected_candidate_ids=["dataset-candidate-1"],
    )

    with pytest.raises(ExperimentPlanningError, match="dataset_selection_request_mismatch"):
        planner.propose_experiment(diagnosis, intervention, selection, None)


def test_blocking_selection_issue_prevents_experiment() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    intervention = _proposal(planner, diagnosis, "replace_or_mix_dataset")
    request, _ = planner.create_discovery_requests(diagnosis, intervention)
    assert request is not None
    selection = DatasetSelectionDecision(
        decision_id="dataset-decision-blocked",
        request_id=request.request_id,
        status="selected",
        selected_candidate_ids=["dataset-candidate-1"],
        issues=[ContractIssue("license", "license_unknown", "License unresolved.", blocking=True)],
    )

    with pytest.raises(ExperimentPlanningError, match="dataset_selection_contains_blocking_issue"):
        planner.propose_experiment(diagnosis, intervention, selection, None)


def test_baseline_is_required_unless_explicitly_justified() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    diagnosis.metadata.pop("baseline_ref")
    intervention = _proposal(planner, diagnosis, "change_training_recipe")

    with pytest.raises(ExperimentPlanningError, match="baseline_required"):
        planner.propose_experiment(diagnosis, intervention, None, None)

    diagnosis.metadata["baseline_justification"] = "No prior checkpoint exists for a controlled from-scratch smoke test."
    intervention = _proposal(planner, diagnosis, "change_training_recipe")
    experiment = planner.propose_experiment(diagnosis, intervention, None, None)
    assert experiment.baseline_ref is None
    assert experiment.metadata["baseline_justification"]
    assert "baseline_absence_justification" in experiment.required_evidence
    assert "baseline_comparison" not in experiment.required_evidence
    assert experiment.success_criteria["baseline_comparison_required"] is False


def test_high_impact_approval_policy_is_preserved() -> None:
    policy = ExperimentPolicy()
    assert policy.required_approvals("rollback", "stabilization", "medium") == (
        "operator_high_risk_approval",
    )
    assert policy.required_approvals("branch", "smoke_test", "medium") == ()


def test_policy_approvals_cannot_be_removed_from_intervention_metadata() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis(
        hypotheses=[
            DiagnosticHypothesis(
                hypothesis_id="hyp-checkpoint",
                failure_domain="checkpoint_integrity",
                summary="The current checkpoint is corrupt.",
                recommended_intervention_kinds=["rollback"],
                confidence=0.9,
            )
        ]
    )
    intervention = _proposal(planner, diagnosis, "rollback")
    intervention.metadata["required_approvals"] = []

    experiment = planner.propose_experiment(diagnosis, intervention, None, None)

    assert experiment.required_approvals == ["operator_high_risk_approval"]
    assert experiment.status in CONTRACT_STATUSES
    assert experiment.metadata["approval_state"] == "required"


def test_incomplete_or_blocked_diagnosis_only_returns_diagnostic_work() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    diagnosis.status = "inconclusive"
    diagnosis.issues = [
        ContractIssue("missing-gradients", "missing_evidence", "Gradient evidence is absent.", blocking=True)
    ]

    interventions = list(planner.propose_interventions(diagnosis))

    assert interventions
    assert all(item.intervention_kind in {"collect_more_evidence", "repair_evaluation"} for item in interventions)

    stale_training_intervention = _proposal(ClosedLoopExperimentPlanner(), _diagnosis(), "change_training_recipe")
    with pytest.raises(ExperimentPlanningError, match="diagnosis_not_ready_for_training_experiment"):
        planner.propose_experiment(diagnosis, stale_training_intervention, None, None)


def test_poisoned_lineage_does_not_continue_training() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis(hypotheses=[_diagnosis().hypotheses[0]])
    diagnosis.metadata["lineage_status"] = "poisoned"

    interventions = list(planner.propose_interventions(diagnosis))

    assert all(item.intervention_kind != "change_training_recipe" for item in interventions)
    assert any(
        reason == "known_dead_end_without_new_evidence"
        for item in interventions
        for reason in item.alternatives_rejected.values()
    )


def test_controls_describe_primary_variable_exceptions_without_contradiction() -> None:
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()

    recipe = _proposal(planner, diagnosis, "change_training_recipe")
    dataset = _proposal(planner, diagnosis, "replace_or_mix_dataset")

    assert recipe.controlled_variables["training_recipe"] == "held_constant_except_learning_rate"
    assert "except_mixture" in str(dataset.controlled_variables["dataset_manifest"])


def test_planner_conforms_to_shared_service_protocol() -> None:
    assert isinstance(ClosedLoopExperimentPlanner(), ExperimentPlanningService)
