from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.policy.code_edit_policy import evaluate_code_edit_proposal
from hephaestus.schemas.code_edit_proposal import CodeEditExecutionRecord, CodeEditProposal
from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class CodeEditProposalStore:
    root: Path

    def append(self, proposal: CodeEditProposal | dict[str, object]) -> None:
        evaluated = evaluate_code_edit_proposal(proposal)
        if self.get(evaluated.proposal_id) is not None:
            return
        JsonStore(self.root, "code_edit_proposals.jsonl").append(evaluated.to_dict())

    def get(self, proposal_id: str) -> dict[str, object] | None:
        return JsonStore(self.root, "code_edit_proposals.jsonl").get_latest("proposal_id", proposal_id)

    def list_all(self) -> list[dict[str, object]]:
        latest_by_id: dict[str, dict[str, object]] = {}
        order: list[str] = []
        for row in JsonStore(self.root, "code_edit_proposals.jsonl").all():
            proposal_id = str(row.get("proposal_id") or "")
            if not proposal_id:
                continue
            if proposal_id not in latest_by_id:
                order.append(proposal_id)
            latest_by_id[proposal_id] = row
        return [latest_by_id[proposal_id] for proposal_id in order]

    def list_for_run(self, run_id: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("run_id") or "") == run_id]

    def list_for_lineage(self, lineage_id: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("lineage_id") or "") == lineage_id]

    def list_by_status(self, status: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("status") or "") == status]

    def list_pending(self) -> list[dict[str, object]]:
        return self.list_by_status("approval_required")

    def list_blocked(self) -> list[dict[str, object]]:
        return self.list_by_status("blocked")

    def record_resolution(self, proposal: CodeEditProposal | dict[str, object]) -> None:
        evaluated = evaluate_code_edit_proposal(proposal)
        JsonStore(self.root, "code_edit_proposals.jsonl").append(evaluated.to_dict())

    def record_execution(self, execution: CodeEditExecutionRecord | dict[str, object]) -> None:
        record = execution.to_dict() if isinstance(execution, CodeEditExecutionRecord) else dict(execution)
        JsonStore(self.root, "code_edit_executions.jsonl").append(record)

    def get_execution(self, execution_id: str) -> dict[str, object] | None:
        return JsonStore(self.root, "code_edit_executions.jsonl").get_latest("execution_id", execution_id)

    def list_executions_for_proposal(self, proposal_id: str) -> list[dict[str, object]]:
        return [
            row
            for row in JsonStore(self.root, "code_edit_executions.jsonl").all()
            if str(row.get("proposal_id") or "") == proposal_id
        ]
