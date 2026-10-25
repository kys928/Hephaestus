"""Real local-source discovery and Planner approval lineage, no fake science."""
from __future__ import annotations

import json

from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedOperationRequest, WORKFLOW_RECORD_COLLECTION,
)
from hephaestus.providers.datasets.local_fixture import (
    LocalFixtureDatasetProvider, LocalFixtureDescriptor,
)
from hephaestus.schemas.discovery_contract import DatasetSearchRequest
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.utils.hashing import hash_file

from test_staged_real_training_handoff import _runtime


def _request(step: str):
    return StagedOperationRequest(
        workflow_id="dataset-approval-integrated",
        run_id="local-discovery-run", lineage_id="local-discovery-lineage",
        stage_name="smoke_test", phase=(
            "planner" if step == "approval_requirement_discovery" else "data_acquisition_audit"),
        substep=step, operation_id="operation-" + step, attempt=1,
        input_refs=(), prior_outputs={},
    )


def _save(runtime, req, kind: str, payload: dict):
    record = {
        "record_id": "record-" + req.substep + "-" + kind,
        "workflow_id": req.workflow_id, "run_id": req.run_id,
        "lineage_id": req.lineage_id, "stage_name": req.stage_name,
        "phase": req.phase, "substep": req.substep,
        "operation_id": req.operation_id, "sequence": 1,
        "kind": kind, "payload": payload,
    }
    ref = runtime.artifact_store.put_bytes(
        canonical_bytes(record), media_type="application/json").artifact_ref
    runtime.state_repository.append(
        WORKFLOW_RECORD_COLLECTION, {**record, "artifact_ref": ref})
    return ref


def _setup(tmp_path):
    runtime = _runtime(tmp_path)
    data = tmp_path / "research-test-source.jsonl"
    data.write_text(
        json.dumps({"text": "An authored synthetic research-only dataset example."}) + "\n"
        + json.dumps({"text": "A separate second authored synthetic line."}) + "\n",
        encoding="utf-8",
    )
    runtime.dataset_registry.register(LocalFixtureDatasetProvider(fixtures=(
        LocalFixtureDescriptor(
            path=data, dataset_id="own-authored-local-research-fixture",
            revision="sha256:" + hash_file(data),
            task_types=("data quality",), domains=("language model data quality",),
            license="CC0-1.0", synthetic=True,
        ),
    )))
    runtime.dataset_registry.provider_allowlist = {"local_fixture"}
    request = DatasetSearchRequest(
        request_id="dataset-search-source-pinned", diagnosis_report_id="diagnosis-verified-source",
        problem_statement="data quality needs better evidence",
        capability_targets=["data quality"], provider_allowlist=["local_fixture"],
        metadata={"intervention_id": "intervention-source-pinned"},
        evidence_refs=["sha256:" + "a" * 64],
    )
    _save(runtime, _request("discovery_request_preparation"), "search_request", {
        "dataset_request": request.to_dict(), "model_request": None,
        "discovery_required": True,
    })
    service = runtime.with_discovery_services(
        StagedAutonomousServices(), require_planner_approval_receipt=True)
    return runtime, service, data


def test_genuine_local_discovery_keeps_original_planner_approval_requirements(tmp_path):
    runtime, services, source_file = _setup(tmp_path)
    _save(runtime, _request("approval_requirement_discovery"), "approval_requirements", {
        "intervention_id": "intervention-source-pinned",
        "required_approvals": ["dataset_selection_approval"],
    })
    actual = services.dataset_discovery.execute(_request("dataset_discovery"))
    assert actual.status == "completed"
    assert len(actual.records[0].payload["candidates"]) == 1
    item = actual.records[0].payload["candidates"][0]
    assert item["provenance"]["content_hash"] == "sha256:" + hash_file(source_file)
    assert item["metadata"]["synthetic"]
    _save(runtime, _request("dataset_discovery"),
          "dataset_discovery_result", actual.records[0].payload)
    choice = services.dataset_selection.execute(_request("dataset_selection"))
    assert choice.status == "completed"
    assert choice.records[0].payload["status"] == "selected"
    assert choice.metadata["required_approvals"] == [
        "dataset_acquisition", "dataset_selection_approval"]
    assert choice.metadata["subject_ref"] == choice.records[0].payload["decision_id"]
    # A selection is only a proposal; the independent gate must obtain a
    # matching decision after this point, not infer approval from the score.
    assert not any(row.get("kind") == "approval_decision"
                   for row in runtime.state_repository.all(WORKFLOW_RECORD_COLLECTION))


def test_missing_or_wrong_intervention_approval_record_fails_closed(tmp_path):
    runtime, services, _ = _setup(tmp_path)
    discovered = services.dataset_discovery.execute(_request("dataset_discovery"))
    assert discovered.status == "completed"
    _save(runtime, _request("dataset_discovery"),
          "dataset_discovery_result", discovered.records[0].payload)
    no_ledger = services.dataset_selection.execute(_request("dataset_selection"))
    assert no_ledger.status == "blocked"
    assert "discovery_handoff_invalid:ValueError" in no_ledger.blocking_issues
    _save(runtime, _request("approval_requirement_discovery"), "approval_requirements", {
        "intervention_id": "wrong-diagnosis-experiment",
        "required_approvals": ["dataset_selection_approval"],
    })
    wrong = services.dataset_selection.execute(_request("dataset_selection"))
    assert wrong.status == "blocked"
    assert "discovery_handoff_invalid:ValueError" in wrong.blocking_issues
