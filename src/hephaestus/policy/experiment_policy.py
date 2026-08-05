"""Policy and heuristic ranking for controlled experiment proposals.

The numeric values in this module are bounded heuristics.  They are useful for
stable ordering, not calibrated probabilities or promises of model improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.schemas.experiment_contract import InterventionProposal


PRIMARY_VARIABLES = frozenset(
    {
        "dataset_mixture",
        "data_quality_filter",
        "preprocessing_policy",
        "tokenizer",
        "learning_rate",
        "scheduler",
        "training_duration",
        "batch_construction",
        "model_candidate",
        "checkpoint_resume_point",
        "decoding_setting",
        "diagnostic_measurement",
        "evaluation_protocol",
        "launch_configuration",
        "runtime_environment",
        "termination_decision",
    }
)

_HIGH_IMPACT_ACTIONS = {
    "rollback": "rollback_to_checkpoint",
    "branch": "branch_new_experiment",
    "restart": "restart_lineage",
}

_LOCAL_APPROVALS = {
    "repair_evaluation": ("eval_policy_change_approval",),
    "replace_or_mix_dataset": ("dataset_selection_approval",),
    "change_tokenizer": ("operator_high_risk_approval",),
    "change_model": ("model_selection_approval",),
}

_COST_PROFILES: dict[str, dict[str, float]] = {
    "collect_more_evidence": {"compute": 0.10, "data": 0.05, "storage": 0.05, "evaluation": 0.35, "time": 0.20},
    "repair_evaluation": {"compute": 0.05, "data": 0.05, "storage": 0.05, "evaluation": 0.65, "time": 0.35},
    "repair_data": {"compute": 0.10, "data": 0.55, "storage": 0.30, "evaluation": 0.30, "time": 0.45},
    "replace_or_mix_dataset": {"compute": 0.55, "data": 0.80, "storage": 0.65, "evaluation": 0.55, "time": 0.70},
    "change_preprocessing": {"compute": 0.25, "data": 0.45, "storage": 0.35, "evaluation": 0.40, "time": 0.45},
    "change_tokenizer": {"compute": 0.80, "data": 0.65, "storage": 0.55, "evaluation": 0.65, "time": 0.85},
    "change_training_recipe": {"compute": 0.70, "data": 0.10, "storage": 0.45, "evaluation": 0.55, "time": 0.70},
    "resume_training": {"compute": 0.65, "data": 0.05, "storage": 0.40, "evaluation": 0.50, "time": 0.60},
    "change_model": {"compute": 0.90, "data": 0.25, "storage": 0.80, "evaluation": 0.75, "time": 0.90},
    "rollback": {"compute": 0.10, "data": 0.05, "storage": 0.10, "evaluation": 0.30, "time": 0.20},
    "branch": {"compute": 0.55, "data": 0.10, "storage": 0.50, "evaluation": 0.55, "time": 0.60},
    "restart": {"compute": 1.00, "data": 0.15, "storage": 0.70, "evaluation": 0.80, "time": 1.00},
    "stop": {"compute": 0.00, "data": 0.00, "storage": 0.00, "evaluation": 0.05, "time": 0.05},
}

_BENEFIT = {
    "collect_more_evidence": 0.35,
    "repair_evaluation": 0.60,
    "repair_data": 0.58,
    "replace_or_mix_dataset": 0.68,
    "change_preprocessing": 0.62,
    "change_tokenizer": 0.72,
    "change_training_recipe": 0.70,
    "resume_training": 0.55,
    "change_model": 0.72,
    "rollback": 0.50,
    "branch": 0.62,
    "restart": 0.58,
    "stop": 0.10,
}

_RISK = {
    "collect_more_evidence": 0.05,
    "repair_evaluation": 0.20,
    "repair_data": 0.25,
    "replace_or_mix_dataset": 0.40,
    "change_preprocessing": 0.30,
    "change_tokenizer": 0.80,
    "change_training_recipe": 0.45,
    "resume_training": 0.35,
    "change_model": 0.75,
    "rollback": 0.25,
    "branch": 0.40,
    "restart": 0.85,
    "stop": 0.05,
}

_REVERSIBILITY = {
    "collect_more_evidence": 1.00,
    "repair_evaluation": 0.90,
    "repair_data": 0.85,
    "replace_or_mix_dataset": 0.80,
    "change_preprocessing": 0.90,
    "change_tokenizer": 0.45,
    "change_training_recipe": 0.85,
    "resume_training": 0.80,
    "change_model": 0.70,
    "rollback": 0.95,
    "branch": 0.95,
    "restart": 0.25,
    "stop": 1.00,
}


@dataclass(frozen=True, slots=True)
class InterventionAssessment:
    score: float
    accepted: bool
    rejection_reason: str | None
    score_components: dict[str, float]
    expected_cost: dict[str, object]
    required_approvals: tuple[str, ...]
    estimate_confidence: float


@dataclass(slots=True)
class ExperimentPolicy:
    """Validate interventions and order them with transparent heuristics."""

    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    minimum_actionable_confidence: float = 0.35
    maximum_missing_evidence_for_training: int = 0

    def required_approvals(
        self,
        intervention_kind: str,
        stage_name: str,
        trust_level: str,
    ) -> tuple[str, ...]:
        approvals = list(_LOCAL_APPROVALS.get(intervention_kind, ()))
        action = _HIGH_IMPACT_ACTIONS.get(intervention_kind)
        if action:
            decision = self.approval_policy.decide(action, stage_name, trust_level)
            if decision.required_approval_type != "none":
                approvals.append(decision.required_approval_type)
        return tuple(sorted(set(approvals)))

    def assess(
        self,
        *,
        intervention_kind: str,
        hypothesis_confidence: float,
        information_gain: float,
        evidence_completeness: float,
        baseline_quality: float,
        stage_appropriateness: float,
        prior_attempts: int,
        known_dead_end: bool,
        has_new_evidence: bool,
        missing_evidence_count: int,
        required_approvals: tuple[str, ...],
    ) -> InterventionAssessment:
        costs = dict(_COST_PROFILES.get(intervention_kind, _COST_PROFILES["collect_more_evidence"]))
        cost_score = sum(float(value) for value in costs.values()) / len(costs)
        prior_penalty = min(1.0, prior_attempts / 3.0)
        approval_penalty = min(1.0, len(required_approvals) / 2.0)
        benefit = _BENEFIT.get(intervention_kind, 0.30)
        risk = _RISK.get(intervention_kind, 0.50)
        reversibility = _REVERSIBILITY.get(intervention_kind, 0.50)

        score = (
            0.24 * hypothesis_confidence
            + 0.18 * information_gain
            + 0.18 * benefit
            + 0.08 * reversibility
            + 0.10 * evidence_completeness
            + 0.07 * baseline_quality
            + 0.07 * stage_appropriateness
            - 0.08 * cost_score
            - 0.08 * risk
            - 0.08 * prior_penalty
            - 0.04 * approval_penalty
        )
        if intervention_kind == "collect_more_evidence" and missing_evidence_count:
            score += 0.20
        score = round(max(0.0, min(1.0, score)), 6)

        rejection_reason: str | None = None
        training_like = intervention_kind not in {"collect_more_evidence", "repair_evaluation", "stop"}
        if known_dead_end and not has_new_evidence:
            rejection_reason = "known_dead_end_without_new_evidence"
        elif training_like and hypothesis_confidence < self.minimum_actionable_confidence:
            rejection_reason = "diagnosis_confidence_below_actionable_threshold"
        elif training_like and missing_evidence_count > self.maximum_missing_evidence_for_training:
            rejection_reason = "missing_evidence_requires_diagnostic_intervention"

        estimate_confidence = round(
            max(0.0, min(1.0, 0.55 * evidence_completeness + 0.45 * hypothesis_confidence)),
            6,
        )
        components = {
            "hypothesis_support": round(hypothesis_confidence, 6),
            "information_gain": round(information_gain, 6),
            "expected_benefit": round(benefit, 6),
            "cost_penalty": round(cost_score, 6),
            "risk_penalty": round(risk, 6),
            "reversibility": round(reversibility, 6),
            "evidence_completeness": round(evidence_completeness, 6),
            "baseline_quality": round(baseline_quality, 6),
            "stage_appropriateness": round(stage_appropriateness, 6),
            "prior_attempt_penalty": round(prior_penalty, 6),
            "approval_penalty": round(approval_penalty, 6),
        }
        return InterventionAssessment(
            score=score,
            accepted=rejection_reason is None,
            rejection_reason=rejection_reason,
            score_components=components,
            expected_cost={
                **costs,
                "aggregate_score": round(cost_score, 6),
                "basis": "bounded_heuristic_not_measured_cost",
            },
            required_approvals=required_approvals,
            estimate_confidence=estimate_confidence,
        )

    def validate_intervention(self, intervention: InterventionProposal) -> list[str]:
        reasons: list[str] = []
        primary = intervention.primary_variable.strip()
        if primary not in PRIMARY_VARIABLES:
            reasons.append("primary_variable_must_name_exactly_one_supported_variable")
        if primary in intervention.controlled_variables:
            reasons.append("primary_variable_cannot_also_be_controlled")
        changed = intervention.metadata.get("changed_variables", [primary])
        if not isinstance(changed, list) or [str(item) for item in changed] != [primary]:
            reasons.append("one_primary_variable_rule_violated")
        if not intervention.controlled_variables:
            reasons.append("controlled_variables_required")
        if not intervention.success_criteria:
            reasons.append("success_criteria_required")
        if not intervention.failure_criteria:
            reasons.append("failure_criteria_required")
        if not intervention.rollback_plan and intervention.intervention_kind != "stop":
            reasons.append("rollback_plan_required")
        return reasons
