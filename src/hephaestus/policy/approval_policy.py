from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hephaestus.config_loader import ConfigError, load_named_config


ALLOWED_APPROVAL_OUTCOMES = {
    "auto_allowed",
    "approval_required",
    "approval_required_high_risk",
    "override_not_allowed",
}


@dataclass(slots=True)
class ApprovalOutcomeResolution:
    outcome: str
    status: str
    effect_on_action: str
    is_override: bool
    override_blocked: bool
    reason: str


@dataclass(slots=True)
class ApprovalGateDecision:
    outcome: str
    risk_level: str
    required_approval_type: str
    reason: str


@dataclass(slots=True)
class ApprovalPolicy:
    config_dir: Path = Path("configs")
    action_rules: dict[str, str] = field(default_factory=dict)
    high_risk_actions: set[str] = field(default_factory=set)
    trust_level_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    stage_overrides: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_rules:
            payload = load_named_config(self.config_dir, "policies", "approval_policy")
            self.action_rules = {str(k): str(v) for k, v in dict(payload.get("action_rules", {})).items()}
            self.high_risk_actions = {str(v) for v in list(payload.get("high_risk_actions", []))}
            self.trust_level_overrides = {
                str(k): {str(a): str(b) for a, b in dict(v).items()}
                for k, v in dict(payload.get("trust_level_overrides", {})).items()
            }
            self.stage_overrides = {
                str(k): {str(a): str(b) for a, b in dict(v).items()}
                for k, v in dict(payload.get("stage_overrides", {})).items()
            }
        self._validate()

    def _validate(self) -> None:
        for table in [self.action_rules, *self.trust_level_overrides.values(), *self.stage_overrides.values()]:
            for action, outcome in table.items():
                if outcome not in ALLOWED_APPROVAL_OUTCOMES:
                    raise ConfigError(
                        f"approval policy outcome for action '{action}' must be one of {sorted(ALLOWED_APPROVAL_OUTCOMES)}"
                    )

    def decide(
        self,
        action: str,
        stage_name: str,
        trust_level: str,
    ) -> ApprovalGateDecision:
        outcome = self.action_rules.get(action, "auto_allowed")
        outcome = self.trust_level_overrides.get(trust_level, {}).get(action, outcome)
        outcome = self.stage_overrides.get(stage_name, {}).get(action, outcome)

        risk = "low"
        if action in self.high_risk_actions:
            risk = "high"
        if outcome == "approval_required_high_risk":
            risk = "high"
        elif outcome == "approval_required":
            risk = "moderate"

        required = "none"
        if outcome == "approval_required":
            required = "operator_approval"
        if outcome == "approval_required_high_risk":
            required = "operator_high_risk_approval"
        if outcome == "override_not_allowed":
            required = "operator_approval_no_override"

        return ApprovalGateDecision(
            outcome=outcome,
            risk_level=risk,
            required_approval_type=required,
            reason=f"approval_policy:{action}:{outcome}",
        )

    def resolve_operator_outcome(self, outcome: str, *, override_allowed: bool) -> ApprovalOutcomeResolution:
        normalized = str(outcome).strip() or "rejected"
        allowed = {"approved", "rejected", "expired", "superseded", "override_approved", "override_rejected"}
        if normalized not in allowed:
            normalized = "rejected"
            return ApprovalOutcomeResolution(
                outcome=normalized,
                status="rejected",
                effect_on_action="hold_requested_action",
                is_override=False,
                override_blocked=False,
                reason="invalid_operator_outcome",
            )

        is_override = normalized.startswith("override_")
        if is_override and not override_allowed:
            return ApprovalOutcomeResolution(
                outcome=normalized,
                status="rejected",
                effect_on_action="hold_requested_action",
                is_override=True,
                override_blocked=True,
                reason="override_not_allowed_by_policy",
            )

        status = "approved" if normalized in {"approved", "override_approved"} else normalized.replace("override_", "")
        effect = "execute_requested_action" if status == "approved" else "hold_requested_action"
        return ApprovalOutcomeResolution(
            outcome=normalized,
            status=status,
            effect_on_action=effect,
            is_override=is_override,
            override_blocked=False,
            reason="operator_outcome_applied",
        )
