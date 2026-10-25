"""Remote IO and scientific attestations here are explicit test fixtures.

Actual staged adapters, cache, streamed byte checks, preprocessing, immutable
records and native approval gate execute. These tests are NOT live corpus
clearance, real frozen-tokenizer measurements or a 45-operation campaign.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hephaestus.control.spine import SpinePhase
from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedAutonomousDependencies, StagedWorkflowState, StagedStepState,
)
from hephaestus.data.reviewed_selection import ReviewBoundDatasetSelector
from hephaestus.data.selection import DeterministicDatasetSelectionService
from hephaestus.data.schema_mapping import PinnedDatasetRecordMapper
from hephaestus.policy.dataset_review import (
    AUDITS, REFERENCES, IndependentDatasetReviewPolicy, candidate_sha256,
    processing_profile_sha256, preview_issues,
)
from hephaestus.schemas.dataset_review import DatasetSourceReview, DatasetScreeningAccessScope
from hephaestus.schemas.dataset_mapping import DatasetRecordMapping
from hephaestus.schemas.discovery_contract import DatasetSearchRequest
from hephaestus.storage.verified import canonical_bytes
from test_staged_real_training_handoff import _runtime
from test_staged_approved_local_data_factory import _write_row, _req
from test_production_data_acquisition import (
    FakeRemoteProvider, FakeTransport, _snapshot, _file, _candidate, _selection, DATA, URL, REVISION,
)


class FixtureTokenizerChecker:
    tokenizer_ref = "sha256:" + "1" * 64
    checker_id = "explicit-test-only-tokenizer-checker"

    def check(self, texts):
        return True, {"fixture_only": True}


def _put(rt, payload):
    return rt.artifact_store.put_bytes(canonical_bytes(payload), media_type="application/json").artifact_ref


def _setup(tmp_path, *, trusted=True, review_changes=None):
    rt = _runtime(tmp_path)
    rt.data_preprocessor.tokenizer_checker = FixtureTokenizerChecker()
    candidate = _candidate(provider_id="huggingface", revision=REVISION,
        languages=["en"], task_types=["text-generation"], trust_level="external_metadata",
        metadata={"acquisition_files": ["data/train.jsonl"]})
    digest = candidate_sha256(candidate)
    refs = {name: _put(rt, {"test_fixture_reference": name}) for name in REFERENCES}
    access = DatasetScreeningAccessScope(digest, ["data/train.jsonl"], 1024 * 1024,
        sorted(AUDITS), refs, status="authorized", authorized_by="independent-test-fixture")
    access_ref = _put(rt, access.to_dict())
    hashes = {"data/train.jsonl": "sha256:" + _file().provider_hash}
    audits = {name: _put(rt, {"audit": name, "status": "pass", "coverage": "full_corpus",
        "method": "explicit synthetic attestation fixture, not a scientific audit",
        "candidate_sha256": digest, "source_file_hashes": hashes,
        "reference_manifest_refs": refs, "access_scope_ref": access_ref}) for name in AUDITS}
    review = DatasetSourceReview(digest, "independent-fixture", "full_corpus", True,
        "pass", "pass", "pass", "cleared", [_put(rt, {"test_fixture_rights": True})],
        FixtureTokenizerChecker.tokenizer_ref, "2" * 64, 1000, 0, "test-normalization",
        source_file_hashes=hashes, audit_refs=audits, reference_manifest_refs=refs,
        access_scope_ref=access_ref, processing_profile_sha256=processing_profile_sha256(rt.data_preprocessor))
    for key, value in (review_changes or {}).items():
        setattr(review, key, value)
    review_ref = _put(rt, review.to_dict())
    policy = IndependentDatasetReviewPolicy(rt.artifact_store, FixtureTokenizerChecker.tokenizer_ref,
        "2" * 64, frozenset({review_ref}) if trusted else frozenset(), frozenset({access_ref}))
    selection = _selection(candidate, required_approvals=["dataset_acquisition",
        "dataset_selection_approval", "dataset_review:" + review_ref],
        metadata={"dataset_review_ref": review_ref, "candidate_sha256": digest})
    _write_row(rt, "dataset_discovery", "dataset_discovery_result", {
        "status": "discovered", "request_id": selection.request_id,
        "candidates": [candidate.to_dict()]})
    _write_row(rt, "dataset_selection", "dataset_selection_decision", selection.to_dict())
    _write_row(rt, "acquisition_approval_gate", "approval_decision", {
        "subject_ref": selection.decision_id, "status": "approved",
        "approval_ref": "approval://local-data", "requirements": selection.required_approvals})
    provider = FakeRemoteProvider(_snapshot(provider_id="huggingface"), (_file(),))
    transport = FakeTransport({URL: DATA})
    rt.dataset_acquisition.providers = {"huggingface": provider}
    rt.dataset_acquisition.transport = transport
    service = rt.with_huggingface_data_factory(StagedAutonomousServices(),
        review_policy=policy, acquisition_enabled=True).dataset_acquisition
    return rt, service, candidate, selection, review, review_ref, transport


def test_remote_staged_receipt_to_real_preprocessing_without_relabeling(tmp_path):
    rt, service, candidate, selection, review, ref, transport = _setup(tmp_path)
    assert service.approval_readiness(_req("acquisition_approval_gate")) == ()
    for index, step in enumerate(("dataset_acquisition", "license_provenance_evidence",
            "dataset_audit", "manifest_production"), 3):
        result = service.execute(_req(step))
        assert result.status == "completed", result.blocking_issues
        for record in result.records:
            _write_row(rt, step, record.kind, record.payload, index)
    assert len(transport.requests) == 1
    manifest = result.records[0].payload
    assert manifest["metadata"]["provider_id"] == "huggingface"
    assert manifest["datasets"][0]["version"] == REVISION
    for step in ("approved_source_preprocessing", "deduplication_contamination",
                 "tokenizer_compatibility", "trainable_data_contract"):
        assert service.execute(_req(step)).status == "completed"
    assert len(transport.requests) == 1  # materialization never redownloads


@pytest.mark.parametrize("changes", [
    {"scope": "public_preview"}, {"rights_status": "declared_only"},
    {"unknown_tokens": 11}, {"unknown_tokens": True}, {"total_tokens": 0},
    {"tokenizer_file_sha256": "3" * 64}, {"revision_verified": False},
    {"source_file_hashes": {}}, {"audit_refs": {}}, {"reference_manifest_refs": {}},
    {"access_scope_ref": "sha256:" + "4" * 64}, {"processing_profile_sha256": "sha256:" + "5" * 64},
])
def test_incomplete_or_mismatched_science_blocks_before_network(tmp_path, changes):
    _, service, _, _, _, _, transport = _setup(tmp_path, review_changes=changes)
    assert service.approval_readiness(_req("acquisition_approval_gate"))
    assert service.execute(_req("dataset_acquisition")).status == "blocked"
    assert not transport.requests


def test_candidate_cannot_self_assert_trusted_review(tmp_path):
    _, service, candidate, _, _, _, transport = _setup(tmp_path, trusted=False)
    candidate.metadata["independently_approved"] = True
    assert service.execute(_req("dataset_acquisition")).status == "blocked"
    assert not transport.requests


def test_changed_provider_snapshot_or_file_hash_blocks_before_transfer(tmp_path):
    rt, service, _, _, _, _, transport = _setup(tmp_path)
    provider = rt.dataset_acquisition.providers["huggingface"]
    provider.files = (_file(provider_hash="9" * 64),)
    assert service.execute(_req("dataset_acquisition")).status == "blocked"
    assert not transport.requests


def test_cached_byte_tampering_blocks_manifest(tmp_path):
    rt, service, _, _, _, _, _ = _setup(tmp_path)
    result = service.execute(_req("dataset_acquisition"))
    for record in result.records:
        _write_row(rt, "dataset_acquisition", record.kind, record.payload, 3)
    receipt = next(r.payload for r in result.records if r.kind == "acquisition_receipt")
    from pathlib import Path
    Path(receipt["acquired_files"][0]["cache_ref"]).write_bytes(b"modified")
    assert service.execute(_req("manifest_production")).status == "blocked"


def test_gate_cannot_complete_with_missing_review_even_with_permissive_approval(tmp_path):
    rt, service, _, _, _, _, transport = _setup(tmp_path, trusted=False)
    class NeverCalled:
        def decision_for(self, request):
            raise AssertionError("an approval must not override scientific eligibility")
    deps = StagedAutonomousDependencies(services=StagedAutonomousServices(dataset_acquisition=service),
        state_repository=rt.state_repository, artifact_store=rt.artifact_store, approval_service=NeverCalled())
    orchestrator = build_staged_autonomous_orchestrator(workflow_id="wf-local-data", run_id="data-run",
        lineage_id="lineage-1", stage_name="smoke_test", dependencies=deps)
    state = StagedWorkflowState("wf-local-data", "data-run", "lineage-1", "smoke_test", steps={
        "data_acquisition_audit.dataset_selection": StagedStepState("data_acquisition_audit",
            "dataset_selection", "op-selection", completion_marker=True, output={
                "required_approvals": ["dataset_acquisition"], "subject_ref": "selection",
                "remote_dataset_review_required": True})})
    result = orchestrator._approval_gate(state, SpinePhase.DATA_ACQUISITION_AUDIT,
        "acquisition_approval_gate", "op-approval")
    assert result.status == "approval_pending"
    assert "independent_full_corpus_review_required" in result.blocking_issues
    assert not transport.requests


def test_review_changes_invalidate_previous_selection_identity(tmp_path):
    rt, _, candidate, _, review, ref, _ = _setup(tmp_path)
    # Test selector mechanics with deterministic admissible evidence, not live rights.
    selector = ReviewBoundDatasetSelector(DeterministicDatasetSelectionService(minimum_score=0),
        IndependentDatasetReviewPolicy(rt.artifact_store, review.tokenizer_ref, review.tokenizer_file_sha256),
        {candidate.candidate_id: ref})
    request = DatasetSearchRequest("search", "diagnosis", "text generation", required_languages=["en"])
    one = selector.select(request, [candidate])
    assert one.status == "selected"
    review.reviewer_id = "a different independent fixture reviewer"
    selector.review_refs[candidate.candidate_id] = _put(rt, review.to_dict())
    two = selector.select(request, [candidate])
    assert one.decision_id != two.decision_id
    assert one.required_approvals != two.required_approvals


def test_json_mapping_preserves_source_and_schema_and_never_uses_feedback():
    mapper = PinnedDatasetRecordMapper(DatasetRecordMapping(["instruction", "data", "schema"], "output"))
    row = {"instruction": "Return JSON", "data": "Order 123", "schema": '{"type":"object"}',
        "output": '{"order":123}', "feedback": "Do not train this"}
    mapped = mapper.map(row)
    assert all(row[key] in mapped["prompt"] for key in ("instruction", "data", "schema"))
    assert mapped["target"] == row["output"]
    assert "feedback" not in str(mapped)
    del row["data"]
    assert mapper.map(row) is None


def test_review_unknown_threshold_stays_one_percent_with_exact_counts(tmp_path):
    _, _, candidate, _, review, _, _ = _setup(tmp_path)
    review.total_tokens, review.unknown_tokens = 1000, 10
    assert preview_issues(review, candidate, tokenizer_ref=review.tokenizer_ref,
        tokenizer_sha256=review.tokenizer_file_sha256) == []
    review.unknown_tokens = 11
    assert "dataset_unknown_token_limit_exceeded" in preview_issues(review, candidate,
        tokenizer_ref=review.tokenizer_ref, tokenizer_sha256=review.tokenizer_file_sha256)
