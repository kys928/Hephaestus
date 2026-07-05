from __future__ import annotations

from hephaestus.schemas.operator_action import ALLOWED_OPERATOR_ACTIONS, OperatorAction


def evaluate_operator_action(action: OperatorAction | dict[str, object]) -> OperatorAction:
    candidate = action if isinstance(action, OperatorAction) else OperatorAction.from_dict(dict(action))
    metadata = dict(candidate.metadata)
    reasons: list[str] = []
    status = "accepted"

    if candidate.action_type not in ALLOWED_OPERATOR_ACTIONS:
        reasons.append("unsupported_action_type")
    if candidate.action_type in {"approve_code_edit", "reject_code_edit"}:
        if candidate.target_type != "code_edit_proposal":
            reasons.append("code_edit_action_requires_code_edit_proposal_target")
        if not candidate.target_id:
            reasons.append("target_id_required")
    if candidate.action_type == "note" and not candidate.reason:
        reasons.append("note_reason_required")

    if reasons:
        status = "rejected"
    metadata["policy_reasons"] = reasons
    return OperatorAction.from_dict({**candidate.to_dict(), "status": status, "metadata": metadata})
