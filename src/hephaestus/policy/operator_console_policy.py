from __future__ import annotations

from dataclasses import asdict, dataclass, field

from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.policy.approval_policy import ApprovalPolicy


@dataclass(slots=True)
class OperatorConsolePolicy:
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    run_commands: set[str] = field(default_factory=lambda: {"abort_run", "rerun_same_config", "request_recheck", "reject_candidate", "reject_checkpoint"})

    def decide_mutation(self, *, action: str, stage_name: str, trust_level: str, approval_status: str = "") -> dict[str, object]:
        boundary = evaluate_action_boundary(action, {"approval_status": approval_status})
        gate = self.approval_policy.decide(action=action, stage_name=stage_name, trust_level=trust_level)
        allowed = bool(boundary.get("allowed", False))
        return {
            "allowed": allowed,
            "action_boundary": boundary,
            "approval_gate": asdict(gate),
            "reason": "allowed" if allowed else ";".join(str(item) for item in boundary.get("reasons", [])),
        }

    def decide_run_command(self, *, command: str, stage_name: str, trust_level: str) -> dict[str, object]:
        if command not in self.run_commands:
            return {"allowed": False, "reason": "unsupported_run_command", "supported_commands": sorted(self.run_commands)}
        return self.decide_mutation(action=command, stage_name=stage_name, trust_level=trust_level, approval_status="approved")
