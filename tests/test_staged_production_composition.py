"""Negative admission tests: binding inventory is not a real-weight campaign."""
from __future__ import annotations

import pytest

from hephaestus.control.staged_state import (
    StagedAutonomousDependencies, StagedAutonomousServices,
    StagedWorkflowState,
)
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.staged_composition import (
    compose_verified_staged_dependencies, staged_binding_inventory,
)
from hephaestus.production.staged_trace import verify_completed_staged_trace
from test_staged_autonomous_orchestration import FakeOperationRouter


def _runtime(tmp_path):
    return ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path / "state", artifact_root=tmp_path / "artifacts",
    )).build()


def test_missing_scientific_services_block_even_when_state_and_approvals_exist(tmp_path):
    rt = _runtime(tmp_path)
    deps = StagedAutonomousDependencies(
        services=StagedAutonomousServices(),
        state_repository=rt.state_repository,
        artifact_store=rt.artifact_store,
        approval_service=object(),
    )
    with pytest.raises(ValueError, match="missing"):
        staged_binding_inventory(deps)


def test_test_double_cannot_masquerade_as_production_scientific_worker(tmp_path):
    rt = _runtime(tmp_path)
    deps = StagedAutonomousDependencies(
        services=StagedAutonomousServices(judge_entry=FakeOperationRouter()),
        state_repository=rt.state_repository,
        artifact_store=rt.artifact_store,
        approval_service=object(),
    )
    with pytest.raises(ValueError, match="simulated"):
        staged_binding_inventory(deps)


def test_one_path_composition_refuses_missing_core_roles_and_approval(tmp_path):
    rt = _runtime(tmp_path)
    with pytest.raises(ValueError, match="approval"):
        compose_verified_staged_dependencies(
            rt, core_services=StagedAutonomousServices(),
            approval_service=None, reasoner=object(),
            generation_backend=object(),
            baseline_run_artifact_ref="sha256:" + "a" * 64,
            approved_recipe_refs=("sha256:" + "b" * 64,),
            approved_baseline_refs=("frozen-parent",),
        )


def test_trace_cannot_be_minted_for_empty_forged_complete_state(tmp_path):
    rt = _runtime(tmp_path)
    fake = StagedWorkflowState(
        workflow_id="forged", run_id="fake-run",
        lineage_id="lineage", stage_name="smoke_test",
        status="completed", completion_marker=True,
    )
    with pytest.raises(ValueError, match="45-operation"):
        verify_completed_staged_trace(fake, rt.state_repository, rt.artifact_store)
