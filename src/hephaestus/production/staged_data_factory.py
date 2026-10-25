"""Approved, hash-verified local datasets -> existing real data preprocessing.

Only an allowlisted local_fixture candidate with independent staged approval
can be acquired. No network download, guessed dataset, fabricated audit, or
training input. The original data-preprocessing service owns transformations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
)
from hephaestus.data.acquisition import (
    DatasetAcquisitionApproval, acquire_approved_local_candidate,
)
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision
from hephaestus.storage.base import StateRepository, ArtifactStore
from hephaestus.utils.hashing import hash_file

from .staged_records import verified_workflow_payload


_SOURCE_STEPS = frozenset({
    "dataset_acquisition", "license_provenance_evidence", "dataset_audit",
    "manifest_production",
})
_PROCESS_STEPS = frozenset({
    "approved_source_preprocessing", "deduplication_contamination",
    "tokenizer_compatibility", "trainable_data_contract",
})


@dataclass(slots=True)
class StagedApprovedLocalDataFactory:
    repository: StateRepository
    artifacts: ArtifactStore
    preprocessor: object
    maximum_source_bytes: int
    acquisition_enabled: bool = False

    def _selection(self, req):
        discovered = verified_workflow_payload(
            self.repository, self.artifacts, req,
            substep="dataset_discovery", kind="dataset_discovery_result")
        selected = verified_workflow_payload(
            self.repository, self.artifacts, req,
            substep="dataset_selection", kind="dataset_selection_decision")
        if discovered.get("status") != "discovered":
            raise ValueError("no actual dataset discovery was completed")
        decision = DatasetSelectionDecision.from_dict(selected)
        if (decision.status != "selected" or
                len(decision.selected_candidate_ids) != 1 or
                decision.request_id != discovered.get("request_id")):
            raise ValueError("exactly one policy-selected dataset is required")
        candidates = [DatasetCandidate(**row) for row in discovered["candidates"]]
        target = decision.selected_candidate_ids[0]
        matches = [c for c in candidates if c.candidate_id == target]
        if len(matches) != 1:
            raise ValueError("selected dataset has no unique discovered source")
        candidate = matches[0]
        if candidate.provider_id != "local_fixture":
            raise ValueError("remote datasets require a separate governed acquisition adapter")
        if not candidate.license or not candidate.provenance.get("content_hash"):
            raise ValueError("source license and pinned content hash are mandatory")
        return decision, candidate

    def _approval(self, req, decision, candidate):
        gate = req.prior_outputs.get("data_acquisition_audit.acquisition_approval_gate", {})
        if (gate.get("approval_status") != "approved"
                or not gate.get("approval_ref")):
            raise PermissionError("operator has not approved selected dataset acquisition")
        record = verified_workflow_payload(
            self.repository, self.artifacts, req,
            substep="acquisition_approval_gate", kind="approval_decision")
        if (record.get("status") != "approved"
                or record.get("approval_ref") != gate["approval_ref"]
                or record.get("subject_ref") != decision.decision_id
                or "dataset_acquisition" not in record.get("requirements", [])):
            raise PermissionError("operator approval is stale, unscoped or mismatched")
        return DatasetAcquisitionApproval(
            selection_decision_id=decision.decision_id,
            approved_candidate_ids=(candidate.candidate_id,),
            approval_refs=(str(gate["approval_ref"]),),
            approved_requirements=tuple(str(v) for v in record["requirements"]),
        )

    def _source(self, req):
        receipt = verified_workflow_payload(
            self.repository, self.artifacts, req,
            substep="dataset_acquisition", kind="acquisition_receipt")
        if receipt.get("schema") != "approved-local-acquisition.v1":
            raise ValueError("local dataset acquisition receipt schema mismatch")
        decision, candidate = self._selection(req)
        approval = self._approval(req, decision, candidate)
        source = acquire_approved_local_candidate(
            candidate, decision, approval, max_bytes=self.maximum_source_bytes)
        if (receipt.get("candidate_id") != candidate.candidate_id
                or receipt.get("selection_decision_id") != decision.decision_id
                or receipt.get("source_content_hash") != source.source_content_hash
                or receipt.get("source_bytes") != source.source_bytes
                or receipt.get("source_ref") != str(source.source_path)
                or receipt.get("approval_refs") != list(source.approval_refs)):
            raise ValueError("approved source differs from immutable acquisition receipt")
        return decision, candidate, approval, source

    def _prepared(self, req):
        result = verified_workflow_payload(
            self.repository, self.artifacts, req,
            substep="manifest_production", kind="data_factory_result")
        if result.get("schema") != "staged-data-factory.v1":
            raise ValueError("prepared-data record schema mismatch")
        processed = Path(str(result.get("processed_dataset_ref") or ""))
        evidence_path = Path(str(result.get("processing_evidence_ref") or ""))
        if not processed.is_file() or not evidence_path.is_file():
            raise ValueError("required trainable data or processing evidence is missing")
        digest = "sha256:" + hash_file(processed)
        if result.get("processed_content_hash") != digest:
            raise ValueError("trainable bytes differ from scientific processing receipt")
        if json.loads(evidence_path.read_text(encoding="utf-8")) != result.get("processing_evidence"):
            raise ValueError("processing provenance changed after preprocessing")
        self._source(req)
        return result

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep not in _SOURCE_STEPS | _PROCESS_STEPS:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=(f"unsupported_data_factory_substep:{req.substep}",))
        if not self.acquisition_enabled:
            return StagedOperationResult(status="approval_pending", resumable=True,
                blocking_issues=("dataset_acquisition_execution_disabled",))
        try:
            if req.substep == "dataset_acquisition":
                decision, candidate = self._selection(req)
                approval = self._approval(req, decision, candidate)
                source = acquire_approved_local_candidate(
                    candidate, decision, approval, max_bytes=self.maximum_source_bytes)
                return StagedOperationResult(
                    records=(StagedOutputRecord("acquisition_receipt", {
                        "schema": "approved-local-acquisition.v1",
                        "selection_decision_id": decision.decision_id,
                        "candidate_id": candidate.candidate_id,
                        "source_ref": str(source.source_path),
                        "source_content_hash": source.source_content_hash,
                        "source_bytes": source.source_bytes,
                        "record_format": source.record_format,
                        "revision": source.revision,
                        "approval_refs": list(source.approval_refs),
                    }),),
                    metadata={"dataset_revision": source.revision,
                              "approved_source_sha256": source.source_content_hash},
                )
            if req.substep in {"license_provenance_evidence", "dataset_audit"}:
                decision, candidate, approval, source = self._source(req)
                kind = ("license_provenance_evidence" if
                        req.substep == "license_provenance_evidence" else "dataset_audit")
                return StagedOperationResult(
                    records=(StagedOutputRecord(kind, {
                        "selection_decision_id": decision.decision_id,
                        "candidate_id": candidate.candidate_id,
                        "license": candidate.license,
                        "source_content_hash": source.source_content_hash,
                        "source_bytes": source.source_bytes,
                        "approved": True,
                        "provenance": candidate.provenance,
                        "scope": "source_file_hash_and_license_only",
                    }),),
                )
            if req.substep == "manifest_production":
                decision, candidate, approval, source = self._source(req)
                result = self.preprocessor.process(
                    run_id=req.run_id, lineage_id=req.lineage_id,
                    stage_name=req.stage_name, candidate=candidate,
                    selection=decision, approval=approval,
                    tokenizer_ref=None,
                )
                processing = dict(result.processing_evidence)
                return StagedOperationResult(
                    records=(
                        StagedOutputRecord("dataset_manifest", result.manifest.to_dict()),
                        StagedOutputRecord("data_factory_result", {
                            "schema": "staged-data-factory.v1",
                            "processed_dataset_ref": processing["processed_dataset_ref"],
                            "processed_content_hash": result.processed_content_hash,
                            "processing_evidence_ref": processing["processing_evidence_ref"],
                            "processing_evidence": processing,
                            "preprocessing_report": result.preprocessing_report.to_dict(),
                            "trainable_data_contract": result.trainable_data_contract.to_dict(),
                            "dataset_manifest": result.manifest.to_dict(),
                        }),
                    ),
                    metadata={"manifest_id": result.manifest.manifest_id,
                              "processed_content_hash": result.processed_content_hash},
                )
            prepared = self._prepared(req)
            if req.substep == "approved_source_preprocessing":
                return StagedOperationResult(records=(StagedOutputRecord(
                    "preprocessing_report", prepared["preprocessing_report"]),))
            if req.substep == "deduplication_contamination":
                evidence = prepared["processing_evidence"]
                return StagedOperationResult(records=(StagedOutputRecord(
                    "deduplication_contamination_evidence", {
                        "deduplication": evidence["deduplication"],
                        "contamination": evidence["contamination"],
                        "processed_content_hash": prepared["processed_content_hash"],
                    }),))
            if req.substep == "tokenizer_compatibility":
                tokenizer = prepared["processing_evidence"]["tokenizer_compatibility"]
                return StagedOperationResult(records=(StagedOutputRecord(
                    "tokenizer_compatibility", tokenizer),),
                    metadata={"tokenizer_status": tokenizer["status"]})
            return StagedOperationResult(records=(StagedOutputRecord(
                "trainable_data_contract", prepared["trainable_data_contract"]),),
                metadata={"processed_content_hash": prepared["processed_content_hash"],
                          "processed_dataset_ref": prepared["processed_dataset_ref"]})
        except (PermissionError, ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=(f"local_data_factory_blocked:{type(exc).__name__}",))
