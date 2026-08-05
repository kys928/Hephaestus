"""Lifecycle state and transition contract for the autonomous experiment loop."""
from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema
from .contract_common import AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

AUTONOMOUS_EXPERIMENT_STATES = (
    "diagnosis_pending", "diagnosed", "intervention_proposed", "discovery_pending",
    "selection_pending", "approval_pending", "data_preparation_pending", "ready_to_train",
    "training", "interrupted", "evaluation_pending", "judgment_pending", "completed",
    "blocked", "failed", "cancelled",
)

ALLOWED_AUTONOMOUS_EXPERIMENT_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "diagnosis_pending": ("diagnosed", "blocked", "failed", "cancelled"),
    "diagnosed": ("intervention_proposed", "blocked", "cancelled"),
    "intervention_proposed": ("discovery_pending", "approval_pending", "blocked", "cancelled"),
    "discovery_pending": ("selection_pending", "blocked", "failed", "cancelled"),
    "selection_pending": ("approval_pending", "data_preparation_pending", "blocked", "cancelled"),
    "approval_pending": ("data_preparation_pending", "ready_to_train", "blocked", "cancelled"),
    "data_preparation_pending": ("ready_to_train", "blocked", "failed", "cancelled"),
    "ready_to_train": ("training", "blocked", "cancelled"),
    "training": ("interrupted", "evaluation_pending", "failed", "cancelled"),
    "interrupted": ("training", "evaluation_pending", "failed", "cancelled"),
    "evaluation_pending": ("judgment_pending", "blocked", "failed"),
    "judgment_pending": ("completed", "discovery_pending", "ready_to_train", "blocked", "failed"),
    "completed": (), "blocked": (), "failed": (), "cancelled": (),
}


def is_allowed_autonomous_experiment_transition(from_state: str, to_state: str) -> bool:
    return to_state in ALLOWED_AUTONOMOUS_EXPERIMENT_TRANSITIONS.get(from_state, ())


@dataclass(slots=True)
class LifecycleTransition(JsonSchema):
    transition_id: str
    entity_kind: str
    entity_id: str
    from_state: str
    to_state: str
    trigger: str
    actor: str
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    approval_ref: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    contract_version: str = AUTONOMOUS_EXPERIMENT_CONTRACT_VERSION

    def is_allowed(self) -> bool:
        return is_allowed_autonomous_experiment_transition(self.from_state, self.to_state)
