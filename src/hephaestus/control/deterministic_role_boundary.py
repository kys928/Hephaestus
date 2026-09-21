"""Deterministic boundary between Hephaestus role-model reasoning and executable state.

Language models may interpret ambiguous scientific evidence, propose hypotheses, and
recommend bounded actions. They must not be trusted to establish machine-verifiable
facts such as whether an approval is valid, an artifact hash matches, a checkpoint
belongs to a lineage, a stage permits an action, a recheck is complete, or an
idempotency key has already reached terminal state.

This module turns those responsibilities into typed deterministic checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from hephaestus.policy.action_registry import canonical_action_name, evaluate_action_boundary


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    ref: str
    kind: str
    run_id: str | None = None
    lineage_id: str | None = None
    state: str = "verified"
    sha256: str | None = None
    observed_sha256: str | None = None
    superseded_by: str | None = None
    facts: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceValidation:
    requested_refs: tuple[str, ...]
    valid_refs: tuple[str, ...]
    failures: tuple[str, ...]
    facts: Mapping[str, object]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class DeterministicRoleContext:
    run_id: str
    lineage_id: str
    stage_name: str
    approval_status: str = "none"
    stage_allowed_actions: tuple[str, ...] = ()
    verified_facts: Mapping[str, object] = field(default_factory=dict)
    terminal_action_keys: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class DeterministicRoleGate:
    role: str
    requested_action: str
    effective_action: str | None
    allowed: bool
    failures: tuple[str, ...]
    verified_facts: Mapping[str, object]
    validated_evidence_refs: tuple[str, ...]


class EvidenceRegistry:
    """Read-only evidence index used to validate model-cited references."""

    def __init__(self, records: Mapping[str, EvidenceRecord]) -> None:
        self._records = dict(records)

    def validate(
        self,
        refs: Sequence[str],
        *,
        run_id: str,
        lineage_id: str,
    ) -> EvidenceValidation:
        failures: list[str] = []
        valid: list[str] = []
        facts: dict[str, object] = {}

        for ref in refs:
            key = str(ref)
            record = self._records.get(key)
            if record is None:
                failures.append(f"unknown_evidence_ref:{key}")
                continue
            if record.run_id and record.run_id != run_id:
                failures.append(f"wrong_run_evidence_ref:{key}")
                continue
            if record.lineage_id and record.lineage_id != lineage_id:
                failures.append(f"wrong_lineage_evidence_ref:{key}")
                continue
            if record.state != "verified":
                failures.append(f"unverified_evidence_ref:{key}:{record.state}")
                continue
            if record.superseded_by:
                failures.append(f"superseded_evidence_ref:{key}:{record.superseded_by}")
                continue
            if (
                record.sha256
                and record.observed_sha256
                and record.sha256 != record.observed_sha256
            ):
                failures.append(f"evidence_hash_mismatch:{key}")
                continue

            valid.append(key)
            for fact_name, value in record.facts.items():
                if fact_name in facts and facts[fact_name] != value:
                    failures.append(f"conflicting_verified_fact:{fact_name}")
                    continue
                facts[fact_name] = value

        return EvidenceValidation(
            requested_refs=tuple(str(x) for x in refs),
            valid_refs=tuple(valid),
            failures=tuple(sorted(set(failures))),
            facts=dict(sorted(facts.items())),
        )


_MACHINE_FACTS = {
    "approval_valid",
    "artifact_hash_match",
    "budget_available",
    "candidate_checkpoint_integrity_valid",
    "candidate_checkpoint_lineage_match",
    "deterministic_gate_passed",
    "evidence_complete",
    "eval_pack_identity_match",
    "metrics_checksum_valid",
    "model_admitted",
    "provenance_valid",
    "recheck_satisfied",
    "stage_action_allowed",
    "tokenizer_identity_match",
    "training_config_identity_match",
}

_REQUIRED_TRUE_BY_ACTION: dict[str, tuple[str, ...]] = {
    "continue_from_checkpoint": (
        "candidate_checkpoint_integrity_valid",
        "candidate_checkpoint_lineage_match",
        "provenance_valid",
    ),
    "promote_checkpoint": (
        "approval_valid",
        "candidate_checkpoint_integrity_valid",
        "candidate_checkpoint_lineage_match",
        "deterministic_gate_passed",
        "evidence_complete",
        "provenance_valid",
        "recheck_satisfied",
        "stage_action_allowed",
    ),
    "rollback_to_checkpoint": (
        "candidate_checkpoint_integrity_valid",
        "candidate_checkpoint_lineage_match",
    ),
    "branch_new_experiment": (
        "candidate_checkpoint_lineage_match",
    ),
}

_MUTATING_ACTIONS = {
    "continue_from_checkpoint",
    "rerun_same_config",
    "rollback_to_checkpoint",
    "branch_new_experiment",
    "restart_lineage",
    "change_stage",
    "modify_training_recipe",
    "modify_data_policy",
    "modify_eval_policy",
    "promote_checkpoint",
    "mark_lineage_stable",
    "mark_lineage_poisoned",
    "quarantine_lineage",
    "archive_lineage",
}


def verified_facts_for_model(
    context: DeterministicRoleContext,
    validation: EvidenceValidation,
) -> dict[str, object]:
    """Return only machine-verifiable facts suitable for role-model context."""

    merged: dict[str, object] = {}
    for source in (context.verified_facts, validation.facts):
        for key, value in source.items():
            if key in _MACHINE_FACTS:
                merged[key] = value

    if context.stage_allowed_actions:
        merged["stage_policy_externalized"] = True
    merged["evidence_registry_validated"] = validation.passed
    return dict(sorted(merged.items()))


def guard_role_output(
    *,
    role: str,
    output: Mapping[str, object],
    context: DeterministicRoleContext,
    evidence_registry: EvidenceRegistry,
) -> DeterministicRoleGate:
    """Validate the executable portion of a role-model proposal.

    The model remains free to reason about hypotheses, trade-offs, uncertainty, and
    scientific interpretation. Any action that contradicts verified machine facts is
    blocked before it can mutate state.
    """

    requested_action = canonical_action_name(
        str(output.get("action") or output.get("next_action") or "").strip()
    )
    refs_obj = output.get("evidence_refs", ())
    refs = (
        tuple(str(x) for x in refs_obj)
        if isinstance(refs_obj, (list, tuple))
        else ()
    )

    validation = evidence_registry.validate(
        refs,
        run_id=context.run_id,
        lineage_id=context.lineage_id,
    )
    failures = list(validation.failures)

    facts: dict[str, object] = dict(context.verified_facts)
    for key, value in validation.facts.items():
        if key in facts and facts[key] != value:
            failures.append(f"conflicting_verified_fact:{key}")
        else:
            facts[key] = value

    if not requested_action:
        failures.append("missing_action")
    else:
        boundary = evaluate_action_boundary(
            requested_action,
            {"approval_status": context.approval_status},
        )
        if not bool(boundary["allowed"]):
            for reason in boundary["reasons"]:
                failures.append(f"action_boundary:{reason}")

        if context.stage_allowed_actions:
            stage_allowed = requested_action in {
                canonical_action_name(x) for x in context.stage_allowed_actions
            }
            facts["stage_action_allowed"] = stage_allowed
            if not stage_allowed:
                failures.append(f"stage_disallows_action:{requested_action}")

        for fact_name in _REQUIRED_TRUE_BY_ACTION.get(requested_action, ()):
            if facts.get(fact_name) is not True:
                failures.append(f"required_verified_fact_not_true:{fact_name}")

        idempotency_key = str(output.get("idempotency_key") or "").strip()
        if (
            requested_action in _MUTATING_ACTIONS
            and idempotency_key
            and idempotency_key in context.terminal_action_keys
        ):
            failures.append(f"terminal_action_already_recorded:{idempotency_key}")

    for hard_fact in (
        "artifact_hash_match",
        "metrics_checksum_valid",
        "eval_pack_identity_match",
        "tokenizer_identity_match",
        "training_config_identity_match",
    ):
        if facts.get(hard_fact) is False:
            failures.append(f"verified_integrity_failure:{hard_fact}")

    failures = sorted(set(failures))
    allowed = not failures
    return DeterministicRoleGate(
        role=str(role),
        requested_action=requested_action,
        effective_action=requested_action if allowed else None,
        allowed=allowed,
        failures=tuple(failures),
        verified_facts=dict(sorted(facts.items())),
        validated_evidence_refs=validation.valid_refs,
    )
