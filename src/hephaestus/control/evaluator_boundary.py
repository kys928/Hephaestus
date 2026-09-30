"""Deterministic projection from Evaluator scientific state to role-local action.

The Evaluator model is responsible for scientific interpretation. It is not trusted
to independently choose the governed action associated with a finite scientific
state. This module projects the model's scientific decision plus machine-verified
facts into the only role-local action Hephaestus will pass downstream.

The model-proposed action is retained as advisory telemetry so disagreement remains
measurable, but it never overrides this projection.
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


def action_for_evaluator_decision(scientific_decision: str) -> str | None:
    """Return the finite role-local policy action for a known scientific state."""

    return DECISION_TO_ACTION.get(str(scientific_decision).strip())


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
