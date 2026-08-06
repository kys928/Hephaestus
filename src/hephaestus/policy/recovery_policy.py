"""Policy defaults and existing-governance checks for autonomous recovery."""

from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.policy.approval_policy import ApprovalPolicy


@dataclass(slots=True)
class RecoveryPolicy:
    """Conservative policy knobs; all limits are deterministic and inspectable."""

    per_operation_retry_limit: int = 2
    per_failure_signature_retry_limit: int = 3
    per_run_retry_limit: int = 5
    per_experiment_retry_limit: int = 8
    per_lineage_retry_limit: int = 12
    global_window_retry_limit: int = 20
    identical_evidence_retry_limit: int = 1
    failed_action_retry_limit: int = 2
    cumulative_cost_limit: float = 100.0
    minimum_recovery_confidence: float = 0.65
    minimum_automatic_confidence: float = 0.72
    backoff_maximum_seconds: int = 300
    deterministic_jitter_max_seconds: int = 3
    backoff_bases: dict[str, int] = field(
        default_factory=lambda: {
            "provider": 5,
            "worker": 2,
            "storage": 10,
            "training": 15,
            "default": 5,
        }
    )
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)

    def limits(self) -> dict[str, int]:
        return {
            "operation": self.per_operation_retry_limit,
            "failure_signature": self.per_failure_signature_retry_limit,
            "run": self.per_run_retry_limit,
            "experiment": self.per_experiment_retry_limit,
            "lineage": self.per_lineage_retry_limit,
            "global_window": self.global_window_retry_limit,
        }

    def assess_registered_action(
        self,
        registry_action: str | None,
        *,
        stage_name: str,
        trust_level: str,
        approval_status: str,
        stage_allowed_actions: set[str] | None,
    ) -> dict[str, object]:
        if registry_action is None:
            return {
                "known_action": True,
                "allowed": False,
                "requires_approval": False,
                "forbidden": False,
                "reasons": ["non_executable_recommendation"],
                "category": "advisory",
            }

        boundary = evaluate_action_boundary(
            registry_action,
            context={"approval_status": approval_status},
        )
        gate = self.approval_policy.decide(registry_action, stage_name, trust_level)
        requires_approval = bool(boundary["requires_approval"]) or gate.outcome in {
            "approval_required",
            "approval_required_high_risk",
            "override_not_allowed",
        }
        has_approval = approval_status in {"approved", "override_approved"}
        stage_exempt = registry_action in {
            "abort_run",
            "continue_from_checkpoint",
            "quarantine_lineage",
            "record_incident",
            "request_recheck",
        }
        stage_allowed = (
            stage_allowed_actions is None
            or registry_action in stage_allowed_actions
            or stage_exempt
        )
        reasons = [str(item) for item in boundary["reasons"]]
        if (
            requires_approval
            and not has_approval
            and "approval_required_missing_or_unapproved" not in reasons
        ):
            reasons.append("approval_required_missing_or_unapproved")
        if not stage_allowed:
            reasons.append("stage_action_not_allowed")
        allowed = (
            bool(boundary["known_action"])
            and not bool(boundary["forbidden"])
            and stage_allowed
            and (not requires_approval or has_approval)
        )
        return {
            **boundary,
            "allowed": allowed,
            "requires_approval": requires_approval,
            "reasons": sorted(set(reasons)),
            "approval_policy_outcome": gate.outcome,
            "approval_policy_risk": gate.risk_level,
            "stage_allowed": stage_allowed,
        }
