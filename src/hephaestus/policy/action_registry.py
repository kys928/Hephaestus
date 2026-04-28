from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType


CATEGORY_AUTO_ALLOWED = "auto_allowed"
CATEGORY_APPROVAL_REQUIRED = "approval_required"
CATEGORY_HIGH_RISK_APPROVAL_REQUIRED = "high_risk_approval_required"
CATEGORY_FORBIDDEN = "forbidden"


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    action_name: str
    category: str
    description: str
    requires_approval: bool
    high_risk: bool
    forbidden: bool
    required_evidence: list[str] = field(default_factory=list)
    forbidden_when: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def _build_definition(action_name: str, category: str, description: str) -> ActionDefinition:
    requires_approval = category in {CATEGORY_APPROVAL_REQUIRED, CATEGORY_HIGH_RISK_APPROVAL_REQUIRED}
    high_risk = category == CATEGORY_HIGH_RISK_APPROVAL_REQUIRED
    forbidden = category == CATEGORY_FORBIDDEN
    return ActionDefinition(
        action_name=action_name,
        category=category,
        description=description,
        requires_approval=requires_approval,
        high_risk=high_risk,
        forbidden=forbidden,
        required_evidence=(
            ["operator_approval_record", "decision_rationale"]
            if category == CATEGORY_APPROVAL_REQUIRED
            else ["operator_high_risk_approval_record", "decision_rationale", "risk_assessment"]
            if category == CATEGORY_HIGH_RISK_APPROVAL_REQUIRED
            else []
        ),
        forbidden_when=(
            ["always"]
            if category == CATEGORY_FORBIDDEN
            else []
        ),
        metadata={"registry_version": "v1"},
    )


_ACTION_DESCRIPTIONS = {
    "observe_state": "Inspect current lineage/run status without mutation.",
    "read_memory": "Read deterministic retrieval memory records.",
    "summarize_run": "Summarize current run evidence and outputs.",
    "summarize_lineage": "Summarize lineage progression and status.",
    "continue_lineage_best": "Continue lineage using current best-known path.",
    "continue_from_checkpoint": "Continue training/evaluation from selected checkpoint.",
    "rerun_same_config": "Rerun with unchanged bounded configuration.",
    "abort_run": "Abort current run due hard safety/runtime stop.",
    "reject_candidate": "Reject the current candidate checkpoint.",
    "reject_checkpoint": "Reject the current candidate checkpoint (legacy vocabulary).",
    "record_incident": "Persist an operational incident report.",
    "request_recheck": "Request additional bounded evaluation evidence.",
    "branch_new_experiment": "Create a new bounded branch experiment from an origin checkpoint.",
    "rollback_to_checkpoint": "Rollback lineage state to a known checkpoint.",
    "restart_lineage": "Restart lineage execution with bounded reset semantics.",
    "change_stage": "Change stage profile or active stage.",
    "modify_training_recipe": "Update training recipe reference/policy.",
    "modify_data_policy": "Update data policy reference/policy.",
    "modify_eval_policy": "Update evaluation policy reference/policy.",
    "promote_checkpoint": "Promote candidate checkpoint into lineage truth.",
    "mark_lineage_stable": "Mark lineage as stable in persistent state.",
    "mark_lineage_poisoned": "Mark lineage as poisoned/unsafe.",
    "quarantine_lineage": "Quarantine lineage from further autonomous operations.",
    "archive_lineage": "Archive lineage from active progression.",
    "delete_artifact_reference": "Delete persisted artifact references.",
    "reset_lineage_state": "Reset lineage state records.",
    "mutate_frozen_eval_pack": "Mutate frozen eval pack content.",
    "silently_rewrite_loader": "Modify critical loaders without explicit governance record.",
    "non_strict_checkpoint_load_in_critical_path": "Use non-strict checkpoint loading in critical path.",
    "edit_forbidden_file": "Edit protected forbidden files.",
    "promote_failed_deterministic_candidate": "Promote candidate that failed deterministic scorecard.",
    "use_unapproved_dataset": "Use dataset without explicit approval/policy alignment.",
    "continue_poisoned_lineage_without_restart_or_branch": "Continue poisoned lineage without required restart/branch safety action.",
    "delete_run_history": "Delete persisted run history.",
    "delete_decision_history": "Delete persisted decision history.",
    "rewrite_memory_without_source_record": "Rewrite memory records without source provenance.",
    "propose_code_edit": "Propose constrained code-edit plan for review only.",
    "approve_code_edit": "Approve a proposed code edit under operator governance.",
    "execute_code_edit": "Execute an approved code edit in a future constrained executor.",
    "execute_unapproved_code_edit": "Execute code edit without required approval.",
    "edit_forbidden_path": "Attempt edits against forbidden path policy.",
    "rewrite_frozen_eval_pack": "Rewrite frozen evaluation pack files.",
}



AUTO_ALLOWED_ACTIONS = [
    "observe_state",
    "read_memory",
    "summarize_run",
    "summarize_lineage",
    "continue_lineage_best",
    "continue_from_checkpoint",
    "rerun_same_config",
    "abort_run",
    "reject_candidate",
    "reject_checkpoint",
    "record_incident",
    "request_recheck",
]

APPROVAL_REQUIRED_ACTIONS = [
    "branch_new_experiment",
    "rollback_to_checkpoint",
    "restart_lineage",
    "change_stage",
    "modify_training_recipe",
    "modify_data_policy",
    "modify_eval_policy",
    "propose_code_edit",
]

HIGH_RISK_APPROVAL_REQUIRED_ACTIONS = [
    "promote_checkpoint",
    "mark_lineage_stable",
    "mark_lineage_poisoned",
    "quarantine_lineage",
    "archive_lineage",
    "delete_artifact_reference",
    "reset_lineage_state",
    "approve_code_edit",
    "execute_code_edit",
]

FORBIDDEN_ACTIONS = [
    "mutate_frozen_eval_pack",
    "silently_rewrite_loader",
    "non_strict_checkpoint_load_in_critical_path",
    "edit_forbidden_file",
    "promote_failed_deterministic_candidate",
    "use_unapproved_dataset",
    "continue_poisoned_lineage_without_restart_or_branch",
    "delete_run_history",
    "delete_decision_history",
    "rewrite_memory_without_source_record",
    "execute_unapproved_code_edit",
    "edit_forbidden_path",
    "rewrite_frozen_eval_pack",
]


_ALIAS_TO_CANONICAL = {
    "reject_checkpoint": "reject_candidate",
}


_REGISTRY: dict[str, ActionDefinition] = {}
for _name in AUTO_ALLOWED_ACTIONS:
    _REGISTRY[_name] = _build_definition(_name, CATEGORY_AUTO_ALLOWED, _ACTION_DESCRIPTIONS[_name])
for _name in APPROVAL_REQUIRED_ACTIONS:
    _REGISTRY[_name] = _build_definition(_name, CATEGORY_APPROVAL_REQUIRED, _ACTION_DESCRIPTIONS[_name])
for _name in HIGH_RISK_APPROVAL_REQUIRED_ACTIONS:
    _REGISTRY[_name] = _build_definition(_name, CATEGORY_HIGH_RISK_APPROVAL_REQUIRED, _ACTION_DESCRIPTIONS[_name])
for _name in FORBIDDEN_ACTIONS:
    _REGISTRY[_name] = _build_definition(_name, CATEGORY_FORBIDDEN, _ACTION_DESCRIPTIONS[_name])

ACTION_REGISTRY: dict[str, ActionDefinition] = dict(_REGISTRY)
ACTION_REGISTRY_VIEW = MappingProxyType(ACTION_REGISTRY)


def canonical_action_name(action_name: str) -> str:
    return _ALIAS_TO_CANONICAL.get(str(action_name), str(action_name))


def evaluate_action_boundary(
    action_name: str,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    ctx = context or {}
    requested = str(action_name)
    canonical = canonical_action_name(requested)
    definition = ACTION_REGISTRY.get(canonical)
    reasons: list[str] = []

    known_action = definition is not None
    if not known_action:
        reasons.append("unknown_action_not_auto_allowed")
        requires_approval = True
        high_risk = False
        forbidden = False
        category = "unknown"
        required_evidence = ["operator_approval_record", "action_registration"]
    else:
        requires_approval = definition.requires_approval
        high_risk = definition.high_risk
        forbidden = definition.forbidden
        category = definition.category
        required_evidence = list(definition.required_evidence)

    if high_risk:
        requires_approval = True
        reasons.append("high_risk_action_requires_approval")

    if forbidden:
        reasons.append("forbidden_action_blocked")

    approval_status = str(ctx.get("approval_status", "")).strip()
    has_approval = approval_status in {"approved", "override_approved"}
    if requires_approval and not has_approval:
        reasons.append("approval_required_missing_or_unapproved")

    allowed = known_action and not forbidden and (not requires_approval or has_approval)

    metadata: dict[str, object] = {
        "canonical_action_name": canonical,
        "requested_action_name": requested,
    }
    if definition is not None:
        metadata.update(dict(definition.metadata))

    return {
        "action_name": requested,
        "known_action": known_action,
        "category": category,
        "allowed": allowed,
        "requires_approval": requires_approval,
        "high_risk": high_risk,
        "forbidden": forbidden,
        "reasons": reasons,
        "required_evidence": required_evidence,
        "metadata": metadata,
    }
