"""Real, verified scientific discovery -> deterministic candidate selection.

Runs only against explicitly configured allowlisted providers. An absent
Planner discovery request means NO change of source/model was proposed; it
never fabricates a candidate or claims a dataset was downloaded. Dataset
acquisition, preprocessing and training remain separate governed operations.
"""
from __future__ import annotations

from dataclasses import dataclass

from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
)
from hephaestus.schemas.discovery_contract import (
    DatasetSearchRequest, DatasetCandidate, DatasetSelectionDecision,
    ModelSearchRequest, ModelCandidate, ModelSelectionDecision,
)
from hephaestus.storage.base import StateRepository, ArtifactStore
from .staged_records import verified_workflow_payload


@dataclass(slots=True)
class StagedDiscoveryAdapter:
    repository: StateRepository
    artifacts: ArtifactStore
    dataset_registry: object
    dataset_selector: object
    model_coordinator: object
    model_selector: object
    require_planner_approval_receipt: bool = False

    def _planner_dataset_approvals(self, request: StagedOperationRequest, search: DatasetSearchRequest) -> list[str]:
        """Reopen the original Planner approval record before selecting data.

        Merely selecting a candidate must never discard the intervention's
        existing approval requirements. The native staged approval gate later
        checks the entire union against an independent approval provider.
        """
        receipt = verified_workflow_payload(
            self.repository, self.artifacts, request,
            substep="approval_requirement_discovery", kind="approval_requirements")
        intervention = search.metadata.get("intervention_id")
        if not isinstance(intervention, str) or not intervention:
            raise ValueError("discovery request has no Planner intervention identity")
        if receipt.get("intervention_id") != intervention:
            raise ValueError("Planner approval record belongs to a different intervention")
        required = receipt.get("required_approvals")
        if (not isinstance(required, list) or
                any(not isinstance(item, str) or not item.strip() for item in required) or
                len(required) != len(set(required))):
            raise ValueError("Planner approval requirements are not a validated list")
        return sorted(required)

    def _search(self, request: StagedOperationRequest, kind: str):
        bundle = verified_workflow_payload(
            self.repository, self.artifacts, request,
            substep="discovery_request_preparation", kind="search_request",
        )
        value = bundle.get(kind + "_request")
        if value is None:
            if bundle.get("discovery_required") is not False and not (
                (kind == "dataset" and bundle.get("model_request") is not None)
                or (kind == "model" and bundle.get("dataset_request") is not None)
            ):
                raise ValueError("discovery request is missing without an explicit no-change decision")
            return None
        if not isinstance(value, dict):
            raise ValueError("discovery search request must be an object")
        result = DatasetSearchRequest(**value) if kind == "dataset" else ModelSearchRequest(**value)
        if not result.request_id or not result.diagnosis_report_id:
            raise ValueError("discovery request is missing scientific source identity")
        return result

    def _prior_candidates(self, request: StagedOperationRequest, kind: str):
        payload = verified_workflow_payload(
            self.repository, self.artifacts, request,
            substep=kind + "_discovery", kind=kind + "_discovery_result",
        )
        original = self._search(request, kind)
        if original is None:
            if payload.get("status") != "not_required":
                raise ValueError("discovery result contradicts no-change planner decision")
            return None, []
        if payload.get("request_id") != original.request_id or payload.get("status") != "discovered":
            raise ValueError("discovery identity/status changed after publication")
        rows = payload.get("candidates")
        if not isinstance(rows, list) or not rows:
            raise ValueError("selected discovery has no genuine candidates")
        factory = DatasetCandidate if kind == "dataset" else ModelCandidate
        candidates = [factory(**v) for v in rows]
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("discovery candidate identities are duplicated")
        return original, candidates

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep not in {
            "dataset_discovery", "dataset_selection", "model_discovery", "model_selection",
        }:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=(f"unsupported_discovery_operation:{request.substep}",))
        try:
            kind = "dataset" if request.substep.startswith("dataset") else "model"
            record_kind = kind + (
                "_discovery_result" if request.substep.endswith("_discovery")
                else "_selection_decision"
            )
            if request.substep.endswith("_discovery"):
                search = self._search(request, kind)
                if search is None:
                    return StagedOperationResult(records=(StagedOutputRecord(
                        record_kind, {"status": "not_required", "kind": kind,
                                      "reason": "planner_requested_no_" + kind + "_change"},
                    ),), metadata={kind + "_discovery_required": False})
                if kind == "dataset":
                    result = self.dataset_registry.discover(search)
                    candidates = list(result.candidates)
                    issues = list(result.issues)
                else:
                    candidates, issues = self.model_coordinator._discover_models(search)
                blocking = [v.code for v in issues if v.blocking]
                payload = {
                    "status": "discovered" if candidates and not blocking else "blocked",
                    "request_id": search.request_id,
                    "kind": kind,
                    "candidates": [c.to_dict() for c in candidates],
                    "issues": [item.to_dict() for item in issues],
                }
                return StagedOperationResult(
                    status="completed" if payload["status"] == "discovered" else "blocked",
                    resumable=False,
                    records=(StagedOutputRecord(record_kind, payload),),
                    blocking_issues=tuple(blocking or (
                        () if candidates else ("discovery_returned_no_candidates",)
                    )),
                    metadata={kind + "_discovery_required": True,
                              kind + "_request_id": search.request_id},
                )
            search, candidates = self._prior_candidates(request, kind)
            if search is None:
                return StagedOperationResult(records=(StagedOutputRecord(
                    record_kind, {"status": "not_required", "kind": kind,
                                  "reason": "planner_requested_no_" + kind + "_change"},
                ),), metadata={"required_approvals": [],
                               kind + "_selection_required": False})
            decision = (
                self.dataset_selector.select(search, candidates)
                if kind == "dataset" else self.model_selector.select(search, candidates)
            )
            if decision.request_id != search.request_id:
                raise ValueError("selected candidate does not match Planner search identity")
            valid_ids = {candidate.candidate_id for candidate in candidates}
            selected_ids = (
                decision.selected_candidate_ids if kind == "dataset" else
                ([decision.selected_candidate_id] if decision.selected_candidate_id else [])
            )
            if not selected_ids or any(x not in valid_ids for x in selected_ids):
                return StagedOperationResult(
                    status="blocked", resumable=False,
                    records=(StagedOutputRecord(record_kind, decision.to_dict()),),
                    blocking_issues=("no_policy_approved_" + kind + "_selection",),
                )
            if decision.status != "selected":
                return StagedOperationResult(
                    status="blocked", resumable=False,
                    records=(StagedOutputRecord(record_kind, decision.to_dict()),),
                    blocking_issues=("selection_not_approved_by_policy",),
                )
            selected = [c for c in candidates if c.candidate_id in selected_ids]
            approvals = list(decision.required_approvals)
            # Selecting a dataset NEVER grants permission to read/prepare it.
            # The existing staged acquisition approval gate must independently
            # authorize that separate action even for a known-license source.
            if kind == "dataset":
                if self.require_planner_approval_receipt:
                    approvals.extend(self._planner_dataset_approvals(request, search))
                approvals = sorted(set([*approvals, "dataset_acquisition"]))
            metadata = {
                "required_approvals": approvals,
                "subject_ref": decision.decision_id,
                kind + "_selection_required": True,
                kind + "_selection_id": decision.decision_id,
            }
            if len(selected) == 1:
                revision = selected[0].revision
                if revision:
                    metadata[kind + "_revision"] = revision
            return StagedOperationResult(
                records=(StagedOutputRecord(record_kind, decision.to_dict()),),
                metadata=metadata,
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return StagedOperationResult(
                status="blocked", resumable=False,
                blocking_issues=(f"discovery_handoff_invalid:{type(exc).__name__}",),
            )
