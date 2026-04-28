from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.policy.code_edit_policy import evaluate_code_edit_proposal
from hephaestus.schemas.code_edit_proposal import CodeEditProposal
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
        return JsonStore(self.root, "code_edit_proposals.jsonl").all()

    def list_for_run(self, run_id: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("run_id") or "") == run_id]

    def list_for_lineage(self, lineage_id: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("lineage_id") or "") == lineage_id]

    def list_by_status(self, status: str) -> list[dict[str, object]]:
        return [row for row in self.list_all() if str(row.get("status") or "") == status]
