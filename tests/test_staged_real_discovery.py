"""Production discovery adapter tests using actual policy selectors and offline providers."""
from __future__ import annotations

import pytest

from hephaestus.control.staged_state import (
    StagedOperationRequest, WORKFLOW_RECORD_COLLECTION,
)
from hephaestus.storage.verified import canonical_bytes
from hephaestus.providers.datasets import FakeDatasetProvider
from hephaestus.providers.models import FakeModelProvider
from test_autonomous_data_discovery import _request as dataset_request, _candidate
from test_governed_model_discovery import _request as model_request
from test_staged_real_training_handoff import _runtime


def _stage_request(substep: str):
    return StagedOperationRequest(
        workflow_id="discovery-workflow",
        run_id="discovery-run",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        phase="data_acquisition_audit",
        substep=substep,
        operation_id="operation-" + substep,
        attempt=1,
        input_refs=(),
        prior_outputs={},
    )


def _record(runtime, payload):
    doc = {
        "record_id": "planner-search-1", "workflow_id": "discovery-workflow",
        "run_id": "discovery-run", "lineage_id": "lineage-1",
        "stage_name": "smoke_test", "phase": "planner",
        "substep": "discovery_request_preparation",
        "operation_id": "planner-op", "sequence": 1,
        "kind": "search_request", "payload": payload,
    }
    ref = runtime.artifact_store.put_bytes(canonical_bytes(doc), media_type="application/json")
    runtime.state_repository.append(WORKFLOW_RECORD_COLLECTION, {
        **doc, "artifact_ref": ref.artifact_ref,
    })


def _save_decision(runtime, request, decision):
    doc = {
        "record_id": "discovery-" + request.substep,
        "workflow_id": request.workflow_id, "run_id": request.run_id,
        "lineage_id": request.lineage_id, "stage_name": request.stage_name,
        "phase": request.phase, "substep": request.substep,
        "operation_id": request.operation_id, "sequence": 2,
        "kind": decision.records[0].kind,
        "payload": decision.records[0].payload,
    }
    ref = runtime.artifact_store.put_bytes(canonical_bytes(doc), media_type="application/json")
    runtime.state_repository.append(WORKFLOW_RECORD_COLLECTION, {
        **doc, "artifact_ref": ref.artifact_ref,
    })


def test_real_offline_allowlisted_dataset_and_model_discovery_to_selection(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.dataset_registry.register(FakeDatasetProvider(
        candidates=(_candidate("c2"), _candidate("c1")),
    ))
    runtime.dataset_registry.provider_allowlist = {"fake"}
    runtime.coordinator.model_providers["fake_models"] = FakeModelProvider()
    _record(runtime, {
        "dataset_request": dataset_request().to_dict(),
        "model_request": model_request(provider_allowlist=["fake_models"]).to_dict(),
        "discovery_required": True,
    })
    from hephaestus.control.staged_state import StagedAutonomousServices
    stage = runtime.with_discovery_services(StagedAutonomousServices())
    dataset = stage.dataset_discovery.execute(_stage_request("dataset_discovery"))
    assert dataset.status == "completed"
    assert [c["candidate_id"] for c in dataset.records[0].payload["candidates"]] == ["c1", "c2"]
    _save_decision(runtime, _stage_request("dataset_discovery"), dataset)
    selection = stage.dataset_selection.execute(_stage_request("dataset_selection"))
    assert selection.status == "completed"
    assert selection.records[0].payload["selected_candidate_ids"] == ["c1"]
    assert selection.metadata["dataset_revision"] == "v1"

    model = stage.model_discovery.execute(_stage_request("model_discovery"))
    assert model.status == "completed"
    assert len(model.records[0].payload["candidates"]) == 2
    _save_decision(runtime, _stage_request("model_discovery"), model)
    chosen = stage.model_selection.execute(_stage_request("model_selection"))
    assert chosen.status == "completed"
    assert chosen.records[0].payload["selected_candidate_id"] == "fake_models:tiny-char-lm@fixture-v1"
    assert chosen.metadata["model_revision"] == "fixture-v1"


def test_planner_no_discovery_request_is_explicit_non_acquisition(tmp_path):
    runtime = _runtime(tmp_path)
    _record(runtime, {
        "dataset_request": None, "model_request": None, "discovery_required": False,
    })
    from hephaestus.control.staged_state import StagedAutonomousServices
    stage = runtime.with_discovery_services(StagedAutonomousServices())
    for step in ("dataset_discovery", "model_discovery"):
        result = stage.dataset_discovery.execute(_stage_request(step))
        assert result.status == "completed"
        assert result.records[0].payload["status"] == "not_required"
        _save_decision(runtime, _stage_request(step), result)
    for step in ("dataset_selection", "model_selection"):
        result = stage.dataset_selection.execute(_stage_request(step))
        assert result.status == "completed"
        assert result.records[0].payload["status"] == "not_required"
        assert not result.metadata["required_approvals"]


def test_unauthorized_or_offline_provider_never_becomes_selected(tmp_path):
    runtime = _runtime(tmp_path)
    _record(runtime, {
        "dataset_request": dataset_request(provider_allowlist=["not-registered"]).to_dict(),
        "model_request": None, "discovery_required": True,
    })
    from hephaestus.control.staged_state import StagedAutonomousServices
    stage = runtime.with_discovery_services(StagedAutonomousServices())
    result = stage.dataset_discovery.execute(_stage_request("dataset_discovery"))
    assert result.status == "blocked"
    assert result.records[0].payload["status"] == "blocked"
    assert not result.records[0].payload["candidates"]


def test_changed_persisted_discovery_record_blocks_selection(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.dataset_registry.register(FakeDatasetProvider(candidates=(_candidate("c1"),)))
    runtime.dataset_registry.provider_allowlist = {"fake"}
    _record(runtime, {
        "dataset_request": dataset_request().to_dict(),
        "model_request": None, "discovery_required": True,
    })
    from hephaestus.control.staged_state import StagedAutonomousServices
    stage = runtime.with_discovery_services(StagedAutonomousServices())
    result = stage.dataset_discovery.execute(_stage_request("dataset_discovery"))
    assert result.status == "completed"
    _save_decision(runtime, _stage_request("dataset_discovery"), result)
    tampered = runtime.state_repository.get_latest(
        WORKFLOW_RECORD_COLLECTION, "record_id", "discovery-dataset_discovery")
    runtime.state_repository.append(WORKFLOW_RECORD_COLLECTION, {
        **tampered, "payload": {"status": "discovered", "candidates": []},
    })
    selection = stage.dataset_selection.execute(_stage_request("dataset_selection"))
    assert selection.status == "blocked"
    assert "discovery_handoff_invalid:ValueError" in selection.blocking_issues
