from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath

from hephaestus.policy.code_edit_policy import evaluate_code_edit_proposal
from hephaestus.schemas.code_edit_proposal import CodeEditExecutionRecord, CodeEditProposal
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_proposal_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"ced-{digest[:16]}"


def _normalize_relative_path(path: str) -> str | None:
    text = str(path).strip().replace("\\", "/")
    if not text:
        return None
    posix = PurePosixPath(text)
    parts = posix.parts
    if posix.is_absolute() or ".." in parts or any(part == "" for part in parts):
        return None
    return posix.as_posix().lstrip("./") or None


def _execution_id(proposal_id: str, *, dry_run: bool) -> str:
    suffix = "dry-run" if dry_run else "run"
    return f"exec-{suffix}-{proposal_id}"


@dataclass(slots=True)
class CodeEditProposalWorkflow:
    """Stage 11 proposal/governance helper; it never mutates target files."""

    store: CodeEditProposalStore

    def create_proposal(
        self,
        *,
        run_id: str,
        lineage_id: str,
        requested_by: str,
        purpose: str,
        target_files: list[str],
        rollback_plan: str,
        test_plan: list[str],
        metadata: dict[str, object] | None = None,
    ) -> CodeEditProposal:
        payload: dict[str, object] = {
            "run_id": run_id,
            "lineage_id": lineage_id,
            "requested_by": requested_by,
            "purpose": purpose,
            "target_files": list(target_files),
            "rollback_plan": rollback_plan,
            "test_plan": list(test_plan),
            "metadata": dict(metadata or {}),
        }
        payload["proposal_id"] = _stable_proposal_id(payload)
        proposal = evaluate_code_edit_proposal(payload)
        self.store.append(proposal)
        stored = self.store.get(proposal.proposal_id)
        return CodeEditProposal.from_dict(stored or proposal.to_dict())

    def approve_proposal(self, proposal_id: str, *, operator_id: str, note: str = "") -> CodeEditProposal:
        proposal = self._load(proposal_id)
        if proposal.status == "blocked" or proposal.risk_level == "forbidden" or proposal.forbidden_files_touched:
            return proposal
        metadata = dict(proposal.metadata)
        metadata["approval_resolution"] = {
            "outcome": "approved",
            "operator_id": operator_id,
            "note": note,
            "created_at": _utc_now(),
        }
        approved = evaluate_code_edit_proposal({**proposal.to_dict(), "status": "approved", "metadata": metadata})
        self.store.record_resolution(approved)
        return approved

    def reject_proposal(self, proposal_id: str, *, operator_id: str, note: str = "") -> CodeEditProposal:
        proposal = self._load(proposal_id)
        if proposal.status == "blocked" or proposal.risk_level == "forbidden" or proposal.forbidden_files_touched:
            return proposal
        metadata = dict(proposal.metadata)
        metadata["approval_resolution"] = {
            "outcome": "rejected",
            "operator_id": operator_id,
            "note": note,
            "created_at": _utc_now(),
        }
        rejected = evaluate_code_edit_proposal({**proposal.to_dict(), "status": "rejected", "metadata": metadata})
        self.store.record_resolution(rejected)
        return rejected

    def execute_dry_run(self, proposal_id: str, *, requested_by: str) -> CodeEditExecutionRecord:
        proposal = self._load(proposal_id)
        if proposal.status != "approved":
            return CodeEditExecutionRecord(
                execution_id=_execution_id(proposal.proposal_id, dry_run=True),
                proposal_id=proposal.proposal_id,
                run_id=proposal.run_id,
                lineage_id=proposal.lineage_id,
                requested_by=requested_by,
                status="refused",
                dry_run=True,
                reason="proposal_not_approved",
                target_files=list(proposal.target_files),
                rollback_plan=proposal.rollback_plan,
                created_at=_utc_now(),
            )
        return CodeEditExecutionRecord(
            execution_id=_execution_id(proposal.proposal_id, dry_run=True),
            proposal_id=proposal.proposal_id,
            run_id=proposal.run_id,
            lineage_id=proposal.lineage_id,
            requested_by=requested_by,
            status="dry_run_ready",
            dry_run=True,
            reason="approved_proposal_dry_run_only_no_files_mutated",
            target_files=list(proposal.target_files),
            rollback_plan=proposal.rollback_plan,
            created_at=_utc_now(),
        )

    def execute_approved(
        self,
        proposal_id: str,
        *,
        requested_by: str,
        changed_files: list[str],
        metadata: dict[str, object] | None = None,
    ) -> CodeEditExecutionRecord:
        proposal = self._load(proposal_id)
        if proposal.status != "approved":
            execution = CodeEditExecutionRecord(
                execution_id=_execution_id(proposal.proposal_id, dry_run=False),
                proposal_id=proposal.proposal_id,
                run_id=proposal.run_id,
                lineage_id=proposal.lineage_id,
                requested_by=requested_by,
                status="refused",
                dry_run=False,
                reason="proposal_not_approved",
                target_files=list(proposal.target_files),
                rollback_plan=proposal.rollback_plan,
                created_at=_utc_now(),
                metadata=dict(metadata or {}),
            )
            self.store.record_execution(execution)
            return execution

        normalized_paths = [_normalize_relative_path(path) for path in changed_files]
        normalized_changed = sorted({path for path in normalized_paths if path})
        if any(path is None for path in normalized_paths):
            return self._record_refused_execution(
                proposal,
                requested_by=requested_by,
                reason="unauthorized_path_access",
                changed_files=changed_files,
                metadata=metadata,
            )

        approved_targets = set(proposal.allowed_files_touched or proposal.target_files)
        unauthorized = [path for path in normalized_changed if path not in approved_targets]
        if unauthorized:
            execution = self._record_refused_execution(
                proposal,
                requested_by=requested_by,
                reason="unauthorized_path_access",
                changed_files=normalized_changed,
                metadata={**dict(metadata or {}), "unauthorized_paths": unauthorized},
            )
            return execution

        execution = CodeEditExecutionRecord(
            execution_id=_execution_id(proposal.proposal_id, dry_run=False),
            proposal_id=proposal.proposal_id,
            run_id=proposal.run_id,
            lineage_id=proposal.lineage_id,
            requested_by=requested_by,
            status="executed",
            dry_run=False,
            reason="approved_proposal_executed",
            target_files=list(proposal.target_files),
            changed_files=normalized_changed,
            rollback_plan=proposal.rollback_plan,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self.store.record_execution(execution)
        self.store.record_resolution(
            evaluate_code_edit_proposal({**proposal.to_dict(), "status": "executed"})
        )
        return execution

    def _record_refused_execution(
        self,
        proposal: CodeEditProposal,
        *,
        requested_by: str,
        reason: str,
        changed_files: list[str],
        metadata: dict[str, object] | None = None,
    ) -> CodeEditExecutionRecord:
        execution = CodeEditExecutionRecord(
            execution_id=_execution_id(proposal.proposal_id, dry_run=False),
            proposal_id=proposal.proposal_id,
            run_id=proposal.run_id,
            lineage_id=proposal.lineage_id,
            requested_by=requested_by,
            status="refused",
            dry_run=False,
            reason=reason,
            target_files=list(proposal.target_files),
            changed_files=list(changed_files),
            rollback_plan=proposal.rollback_plan,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self.store.record_execution(execution)
        return execution

    def _load(self, proposal_id: str) -> CodeEditProposal:
        record = self.store.get(proposal_id)
        if record is None:
            raise ValueError(f"unknown code edit proposal: {proposal_id}")
        return CodeEditProposal.from_dict(record)
