"""Deterministic projections for the Evaluator's governed output boundary.

The Evaluator model owns scientific interpretation. It does not independently own
finite policy consequences that can be derived exactly from that interpretation.
The model-proposed action and primary variable remain observable telemetry, while
Hephaestus projects authoritative action and primary-variable values in code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DECISION_TO_ACTION: dict[str, str] = {
    "scientific_rejection": "reject_candidate",
    "incomplete_evidence": "request_recheck",
    "improved": "continue_from_checkpoint",
    "regressed": "continue_lineage_best",
    "equivalent": "continue_lineage_best",
    "inconclusive": "hold_candidate",
    "recheck_required": "request_recheck",
    "certification_ready": "certify_candidate",
}

DECISION_TO_PRIMARY_VARIABLE: dict[str, str] = {
    "scientific_rejection": "hard_gate_status",
    "incomplete_evidence": "runtime_evidence",
    "improved": "candidate_quality",
    "regressed": "candidate_regression",
    "equivalent": "effect_size",
    "inconclusive": "evaluation_integrity",
    "recheck_required": "variance_risk",
    "certification_ready": "certification_state",
}

_CERTIFICATION_REQUIRED_TRUE = (
    "deterministic_gate_passed",
    "evidence_complete",
    "provenance_valid",
    "recheck_satisfied",
)


@dataclass(frozen=True, slots=True)
class EvaluatorPolicyProjection:
    scientific_decision: str
    model_proposed_action: str
    effective_action: str | None
    model_action_agrees: bool
    machine_override: bool
    blocked: bool
    reasons: tuple[str, ...]
    verified_facts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EvaluatorPrimaryVariableProjection:
    scientific_decision: str
    model_proposed_primary_variable: str
    effective_primary_variable: str | None
    model_primary_variable_agrees: bool
    blocked: bool
    reasons: tuple[str, ...]


def action_for_evaluator_decision(scientific_decision: str) -> str | None:
    """Return the finite role-local policy action for a known scientific state."""

    return DECISION_TO_ACTION.get(str(scientific_decision).strip())


def primary_variable_for_evaluator_decision(scientific_decision: str) -> str | None:
    """Return the authoritative primary variable for a known scientific state."""

    return DECISION_TO_PRIMARY_VARIABLE.get(str(scientific_decision).strip())


def project_evaluator_primary_variable(
    scientific_decision: str,
    *,
    model_proposed_primary_variable: str = "",
) -> EvaluatorPrimaryVariableProjection:
    """Project scientific state into the only authoritative primary variable.

    This projection intentionally does not infer or repair an unknown scientific
    decision. Unknown decisions fail closed. The model-proposed value is retained as
    telemetry only and never becomes authoritative by itself.
    """

    decision = str(scientific_decision).strip()
    proposed = str(model_proposed_primary_variable).strip()
    effective = primary_variable_for_evaluator_decision(decision)
    if effective is None:
        return EvaluatorPrimaryVariableProjection(
            scientific_decision=decision,
            model_proposed_primary_variable=proposed,
            effective_primary_variable=None,
            model_primary_variable_agrees=False,
            blocked=True,
            reasons=(f"unknown_evaluator_scientific_decision:{decision or '<missing>'}",),
        )
    return EvaluatorPrimaryVariableProjection(
        scientific_decision=decision,
        model_proposed_primary_variable=proposed,
        effective_primary_variable=effective,
        model_primary_variable_agrees=bool(proposed and proposed == effective),
        blocked=False,
        reasons=(),
    )


def project_evaluator_action(
    scientific_decision: str,
    verified_facts: Mapping[str, object] | None = None,
    *,
    model_proposed_action: str = "",
) -> EvaluatorPolicyProjection:
    """Project an Evaluator decision into a deterministic role-local action.

    Machine facts have precedence where they are available. In particular, a failed
    deterministic gate forces rejection, incomplete evidence forces a recheck, and
    invalid provenance forces a hold. A model may claim ``certification_ready`` but
    ``certify_candidate`` is emitted only when every required certification fact is
    explicitly true.

    The projection remains intentionally asymmetric: code may conservatively block or
    downgrade an unsafe model state, but it never upgrades a non-certification model
    decision into ``certify_candidate``.
    """

    decision = str(scientific_decision).strip()
    proposed = str(model_proposed_action).strip()
    facts = dict(verified_facts or {})
    base_action = action_for_evaluator_decision(decision)
    reasons: list[str] = []

    if base_action is None:
        return EvaluatorPolicyProjection(
            scientific_decision=decision,
            model_proposed_action=proposed,
            effective_action=None,
            model_action_agrees=False,
            machine_override=False,
            blocked=True,
            reasons=(f"unknown_evaluator_scientific_decision:{decision or '<missing>'}",),
            verified_facts=dict(sorted(facts.items())),
        )

    effective = base_action

    if facts.get("deterministic_gate_passed") is False:
        effective = "reject_candidate"
        reasons.append("verified_hard_gate_failure_forces_rejection")
    elif facts.get("evidence_complete") is False:
        effective = "request_recheck"
        reasons.append("verified_incomplete_evidence_forces_recheck")
    elif facts.get("provenance_valid") is False:
        effective = "hold_candidate"
        reasons.append("verified_invalid_provenance_forces_hold")
    elif decision == "certification_ready":
        missing = [name for name in _CERTIFICATION_REQUIRED_TRUE if facts.get(name) is not True]
        if missing:
            effective = "request_recheck"
            reasons.extend(f"certification_fact_not_true:{name}" for name in missing)

    machine_override = effective != base_action
    agreement = bool(proposed and proposed == effective)
    return EvaluatorPolicyProjection(
        scientific_decision=decision,
        model_proposed_action=proposed,
        effective_action=effective,
        model_action_agrees=agreement,
        machine_override=machine_override,
        blocked=False,
        reasons=tuple(reasons),
        verified_facts=dict(sorted(facts.items())),
    )
