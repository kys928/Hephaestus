"""Review-bound staged HF acquisition, using the existing bounded downloader.

No synthesized approvals, new downloader, fixture relabeling, or certificate.
Screening access is separate from subsequent selected training acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from hephaestus.control.staged_state import StagedOperationResult, StagedOutputRecord
from hephaestus.data.acquisition_models import AcquisitionReceipt, RemoteAcquisitionLimits
from hephaestus.data.materialization import validate_remote_acquisition_for_preprocessing
from hephaestus.policy.dataset_review import candidate_sha256
from hephaestus.storage.verified import read_verified_json
from .staged_data_factory import StagedApprovedLocalDataFactory
from .staged_records import verified_workflow_payload


@dataclass(slots=True)
class StagedApprovedHuggingFaceDataFactory(StagedApprovedLocalDataFactory):
    remote_acquisition: object = None
    review_policy: object = None
    public_eval_pack_path: Path | None = None

    def _validate_candidate(self, candidate):
        files = candidate.metadata.get("acquisition_files")
        if (candidate.provider_id != "huggingface" or
                not re.fullmatch(r"[0-9a-f]{40}", candidate.revision or "") or
                not isinstance(files, list) or len(files) != 1 or
                not isinstance(files[0], str) or
                not files[0].endswith((".jsonl", ".json", ".csv", ".parquet"))):
            raise ValueError("one explicit nonexecutable HF data file at an immutable commit is required")
        # Multi-shard aggregation needs its own manifest builder; never silently
        # preprocess the first shard and call that a full corpus.

    def _clearance(self, decision, candidate):
        ref = decision.metadata.get("dataset_review_ref")
        if (not isinstance(ref, str) or
                decision.metadata.get("candidate_sha256") != candidate_sha256(candidate) or
                "dataset_review:" + ref not in decision.required_approvals):
            raise PermissionError("selection is not bound to independent source review evidence")
        return self.review_policy.require_clearance(candidate, ref,
            preprocessor=self.preprocessor, maximum_bytes=self.maximum_source_bytes)

    def approval_readiness(self, req):
        try:
            decision, candidate = self._selection(req)
            self._clearance(decision, candidate)
            return ()
        except (PermissionError, ValueError, OSError, TypeError, KeyError):
            return ("independent_full_corpus_review_required",)

    def _receipt(self, req):
        payload = verified_workflow_payload(self.repository, self.artifacts, req,
            substep="dataset_acquisition", kind="acquisition_receipt")
        return AcquisitionReceipt.from_dict(payload)

    def _source(self, req):
        decision, candidate = self._selection(req)
        approval = self._approval(req, decision, candidate)
        review = self._clearance(decision, candidate)
        receipt = self._receipt(req)
        if (len(receipt.acquired_files) != 1 or
                receipt.acquired_files[0].relative_path != candidate.metadata["acquisition_files"][0] or
                receipt.license != candidate.license or any(i.blocking for i in receipt.issues)):
            raise ValueError("acquisition receipt differs from accepted source-file scope")
        source = validate_remote_acquisition_for_preprocessing(
            candidate=candidate, selection=decision, approval=approval, receipt=receipt,
            max_bytes=self.maximum_source_bytes)
        if source.source_content_hash != review.source_file_hashes[candidate.metadata["acquisition_files"][0]]:
            raise ValueError("acquired bytes differ from the independently audited corpus")
        return decision, candidate, approval, source

    def _acquisition_result(self, req, decision, candidate, approval):
        review = self._clearance(decision, candidate)
        access = read_verified_json(self.artifacts, review.access_scope_ref)
        result = self.remote_acquisition.plan(candidate=candidate, selection=decision,
            limits=RemoteAcquisitionLimits(max_bytes=access["maximum_bytes"], max_files=1,
                allowed_suffixes=(".jsonl", ".json", ".csv", ".parquet")))
        if result.status != "ready" or result.plan is None:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=tuple(i.code for i in result.issues) or ("remote_acquisition_plan_missing",))
        plan = result.plan
        if (plan.snapshot.resolved_revision != candidate.revision or
                plan.snapshot.dataset_id != candidate.dataset_id or
                plan.snapshot.provider_id != candidate.provider_id or
                plan.snapshot.license != candidate.license or
                {f.relative_path for f in plan.files} != set(review.source_file_hashes)):
            raise ValueError("remote plan differs from reviewed identity, license or file scope")
        for file in plan.files:
            if (file.provider_hash_algorithm == "sha256" and file.provider_hash and
                    "sha256:" + file.provider_hash.removeprefix("sha256:") != review.source_file_hashes[file.relative_path]):
                raise ValueError("provider content hash differs from audited source")
        acquired = self.remote_acquisition.acquire(plan, approval)
        records = [StagedOutputRecord("remote_acquisition_plan", plan.to_dict()),
                   StagedOutputRecord("acquisition_receipt", acquired.receipt.to_dict()),
                   StagedOutputRecord("dataset_scientific_review", {
                       "review_ref": decision.metadata["dataset_review_ref"],
                       "candidate_sha256": candidate_sha256(candidate),
                       "scope": review.scope, "audit_refs": review.audit_refs})]
        if not acquired.completed:
            return StagedOperationResult(status="blocked", records=tuple(records),
                blocking_issues=tuple(i.code for i in acquired.receipt.issues) or ("remote_acquisition_incomplete",))
        source = validate_remote_acquisition_for_preprocessing(candidate=candidate,
            selection=decision, approval=approval, receipt=acquired.receipt,
            max_bytes=self.maximum_source_bytes)
        if source.source_content_hash != review.source_file_hashes[plan.files[0].relative_path]:
            return StagedOperationResult(status="blocked", records=tuple(records),
                blocking_issues=("acquired_source_differs_from_audited_bytes",))
        return StagedOperationResult(records=tuple(records), metadata={
            "dataset_revision": source.revision,
            "approved_source_sha256": source.source_content_hash,
            "dataset_review_ref": decision.metadata["dataset_review_ref"]})

    def _process_candidate(self, req, decision, candidate, approval):
        self._source(req)  # repeat authorization, audit and cache verification
        if self.preprocessor.continuation_contract is not None:
            if self.public_eval_pack_path is None:
                raise PermissionError("exact continuation requires an explicitly pinned public evaluation pack")
            return self.preprocessor.process_reviewed_continuation_acquisition(
                run_id=req.run_id, lineage_id=req.lineage_id, stage_name=req.stage_name,
                candidate=candidate, selection=decision, approval=approval,
                receipt=self._receipt(req), review_policy=self.review_policy,
                review_ref=decision.metadata["dataset_review_ref"],
                public_eval_pack_path=self.public_eval_pack_path)
        return self.preprocessor.process_remote_acquisition(
            run_id=req.run_id, lineage_id=req.lineage_id, stage_name=req.stage_name,
            candidate=candidate, selection=decision, approval=approval,
            receipt=self._receipt(req), tokenizer_ref=self.review_policy.tokenizer_ref)
