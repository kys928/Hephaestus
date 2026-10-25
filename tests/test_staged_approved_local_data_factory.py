"""Real approved local file -> actual preprocessing -> immutable staged contracts.

No fake preprocessing worker. Approval decisions and candidate discovery are
explicit test fixtures; selection/approval scope and byte hashing are exercised
against real source files and production repositories.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedOperationRequest, WORKFLOW_RECORD_COLLECTION,
)
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision
from hephaestus.storage.verified import canonical_bytes
from test_staged_real_training_handoff import _runtime


def _source(tmp_path):
    source = tmp_path / "source-dataset.jsonl"
    source.write_text(
        "".join(json.dumps({"text": sentence}) + "\n" for sentence in (
            "hello verified source row", "another scientific training example",
            "hello verified source row",
        )), encoding="utf-8",
    )
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    return source, digest


def _write_row(runtime, step, kind, payload, ordinal=1):
    row = {
        "record_id": f"row-{step}-{kind}-{ordinal}",
        "workflow_id": "wf-local-data", "run_id": "data-run",
        "lineage_id": "lineage-1", "stage_name": "smoke_test",
        "phase": ("data_preprocessor" if step in (
            "approved_source_preprocessing", "deduplication_contamination",
            "tokenizer_compatibility", "trainable_data_contract") else "data_acquisition_audit"),
        "substep": step, "operation_id": "op-" + step,
        "sequence": ordinal, "kind": kind, "payload": payload,
    }
    ref = runtime.artifact_store.put_bytes(canonical_bytes(row), media_type="application/json")
    runtime.state_repository.append(WORKFLOW_RECORD_COLLECTION, {
        **row, "artifact_ref": ref.artifact_ref,
    })


def _req(step, *, approved=True):
    return StagedOperationRequest(
        workflow_id="wf-local-data", run_id="data-run",
        lineage_id="lineage-1", stage_name="smoke_test",
        phase="data_preprocessor" if step in (
            "approved_source_preprocessing", "deduplication_contamination",
            "tokenizer_compatibility", "trainable_data_contract") else "data_acquisition_audit",
        substep=step, operation_id="op-" + step, attempt=1,
        input_refs=(),
        prior_outputs={"data_acquisition_audit.acquisition_approval_gate": {
            "approval_status": "approved" if approved else "pending",
            "approval_ref": "approval://local-data" if approved else None,
        }},
    )


def _setup(tmp_path, *, approved=True):
    runtime = _runtime(tmp_path)
    source, digest = _source(tmp_path)
    candidate = DatasetCandidate(
        candidate_id="local-data-1", provider_id="local_fixture",
        dataset_id="local/scientific-repair", revision="v1",
        artifact_ref=str(source),
        task_types=["instruction_following"],
        languages=["en"], domains=["scientific-repair"],
        format_profile={"record_format": "jsonl"},
        estimated_rows=3, estimated_bytes=source.stat().st_size,
        license="mit", provenance={"source": "local_fixture", "content_hash": digest},
        trust_level="verified",
        compatibility={"model_compatible": True},
    )
    selected = DatasetSelectionDecision(
        decision_id="selected-1", request_id="search-1", status="selected",
        selected_candidate_ids=["local-data-1"],
        ranked_candidate_ids=["local-data-1"],
        required_approvals=["dataset_acquisition"],
        selection_rationale="verified local source",
    )
    _write_row(runtime, "dataset_discovery", "dataset_discovery_result", {
        "status": "discovered", "kind": "dataset", "request_id": "search-1",
        "candidates": [candidate.to_dict()], "issues": [],
    })
    _write_row(runtime, "dataset_selection", "dataset_selection_decision", selected.to_dict())
    if approved:
        _write_row(runtime, "acquisition_approval_gate", "approval_decision", {
            "request_id": "operator-approval-1", "operation_id": "approval-op",
            "subject_ref": selected.decision_id, "status": "approved",
            "approval_ref": "approval://local-data",
            "requirements": ["dataset_acquisition"],
        })
    return runtime, source


def _factory(runtime, *, enabled=True):
    return runtime.with_local_data_factory(StagedAutonomousServices(),
                                           acquisition_enabled=enabled).dataset_acquisition


def _completed_source(runtime, service):
    for i, step in enumerate((
        "dataset_acquisition", "license_provenance_evidence",
        "dataset_audit", "manifest_production",
    ), start=3):
        result = service.execute(_req(step))
        assert result.status == "completed", (step, result.blocking_issues)
        for record in result.records:
            _write_row(runtime, step, record.kind, record.payload, ordinal=i)
    return result


def test_approved_local_dataset_flows_into_real_trainable_contract(tmp_path):
    runtime, source = _setup(tmp_path)
    service = _factory(runtime)
    result = _completed_source(runtime, service)
    assert result.records[0].kind == "dataset_manifest"
    data = result.records[1].payload
    assert data["schema"] == "staged-data-factory.v1"
    processed = Path(data["processed_dataset_ref"])
    assert processed.is_file()
    assert data["processed_content_hash"] == "sha256:" + hashlib.sha256(processed.read_bytes()).hexdigest()
    assert data["processing_evidence"]["deduplication"]["exact_duplicates_removed"] == 1

    for step, kind in (
        ("approved_source_preprocessing", "preprocessing_report"),
        ("deduplication_contamination", "deduplication_contamination_evidence"),
        ("tokenizer_compatibility", "tokenizer_compatibility"),
        ("trainable_data_contract", "trainable_data_contract"),
    ):
        outcome = service.execute(_req(step))
        assert outcome.status == "completed", (step, outcome.blocking_issues)
        assert outcome.records[0].kind == kind
    assert outcome.records[0].payload["processed_dataset_ref"] == str(processed)


def test_dataset_preparation_disabled_by_default(tmp_path):
    runtime, source = _setup(tmp_path)
    outcome = _factory(runtime, enabled=False).execute(_req("dataset_acquisition"))
    assert outcome.status == "approval_pending"
    assert "dataset_acquisition_execution_disabled" in outcome.blocking_issues


def test_nonexistent_or_unscoped_operator_approval_rejected(tmp_path):
    runtime, source = _setup(tmp_path, approved=False)
    outcome = _factory(runtime).execute(_req("dataset_acquisition"))
    assert outcome.status == "blocked"
    assert "local_data_factory_blocked:PermissionError" in outcome.blocking_issues


def test_source_hash_change_blocks_before_preprocessing(tmp_path):
    runtime, source = _setup(tmp_path)
    service = _factory(runtime)
    first = service.execute(_req("dataset_acquisition"))
    assert first.status == "completed"
    _write_row(runtime, "dataset_acquisition", first.records[0].kind,
               first.records[0].payload, ordinal=3)
    source.write_text('{"text":"modified after acquisition"}\n', encoding="utf-8")
    result = service.execute(_req("manifest_production"))
    assert result.status == "blocked"
    assert "local_data_factory_blocked:ValueError" in result.blocking_issues


def test_tampered_processed_training_bytes_are_not_used(tmp_path):
    runtime, source = _setup(tmp_path)
    service = _factory(runtime)
    result = _completed_source(runtime, service)
    path = Path(result.records[1].payload["processed_dataset_ref"])
    path.write_bytes(b"fake processed rows")
    outcome = service.execute(_req("trainable_data_contract"))
    assert outcome.status == "blocked"
    assert "local_data_factory_blocked:ValueError" in outcome.blocking_issues
