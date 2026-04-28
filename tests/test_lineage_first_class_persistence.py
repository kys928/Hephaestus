from pathlib import Path

from hephaestus.control.branching import create_branch_state
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.restart import create_restart_state
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore


def test_lineage_round_trip_first_class_fields(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    record = LineageState(
        lineage_id="lineage-a",
        parent_lineage_id=None,
        child_lineage_ids=["lineage-a-b1"],
        stage_name="early_pretraining",
        status="promising",
        trust_level="medium",
        origin_run_id="run-origin",
        origin_checkpoint_ref="artifacts/origin.ckpt",
        branch_origin_checkpoint_ref=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
        architecture_contract_ref="contracts/arch.json",
        tokenizer_contract_ref="contracts/tokenizer.json",
        data_policy_ref="contracts/data_policy.json",
        training_recipe_ref="contracts/recipe.json",
        eval_policy_ref="contracts/eval_policy.json",
        loop_index=2,
        run_count=2,
        latest_run_id="run-2",
        best_checkpoint_ref="artifacts/run-2/best.ckpt",
        last_stable_checkpoint_ref="artifacts/run-2/stable.ckpt",
        certified_stable_checkpoint_ref="artifacts/run-2/cert.ckpt",
        last_certification_result="certification_passed",
        last_decision="promote_checkpoint",
        last_decision_id="dec-run-2-exit",
        last_requested_action="promote_checkpoint",
        last_effective_action="promote_checkpoint",
        last_approval_status="approved",
        pending_approval=False,
        last_high_impact_request_id="apr-run-2-promote",
        last_repeated_eval_count=3,
        last_consistency_score=0.95,
        last_variance_risk="low",
        certification_recheck_count=1,
        repeatability_sufficient=True,
        recent_failures=["run-0"],
        known_pathologies=["old_pathology"],
        major_interventions=[{"type": "rollback", "run_id": "run-1", "target_checkpoint_ref": "artifacts/run-1/stable.ckpt"}],
        metadata={"owner": "team-a", "ticket": "HP-42"},
    )
    store.set_current(record.to_dict())

    loaded = store.get_current("lineage-a")

    assert loaded is not None
    assert loaded["lineage_id"] == "lineage-a"
    assert loaded["origin_run_id"] == "run-origin"
    assert loaded["architecture_contract_ref"] == "contracts/arch.json"
    assert loaded["best_checkpoint_ref"] == "artifacts/run-2/best.ckpt"
    assert loaded["last_decision"] == "promote_checkpoint"
    assert loaded["known_pathologies"] == ["old_pathology"]
    assert loaded["major_interventions"][0]["type"] == "rollback"
    assert loaded["metadata"]["ticket"] == "HP-42"


def test_lineage_backwards_compatibility_defaults(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    store.set_current({
        "lineage_id": "legacy-lineage",
        "parent_lineage_id": None,
        "stage_name": "early_pretraining",
        "status": "active",
        "latest_run_id": "legacy-run",
    })

    loaded = store.get_current("legacy-lineage")

    assert loaded is not None
    assert loaded["trust_level"] == "unknown"
    assert loaded["known_pathologies"] == []
    assert loaded["major_interventions"] == []
    assert loaded["metadata"] == {}
    assert loaded["run_count"] == 0


def test_branch_persistence_parent_child_linkage(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    parent = LineageState(
        lineage_id="lineage-main",
        parent_lineage_id=None,
        stage_name="early_pretraining",
        status="stable",
        trust_level="high",
        origin_run_id="run-1",
        origin_checkpoint_ref="artifacts/run-1/stable.ckpt",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        best_checkpoint_ref="artifacts/run-5/best.ckpt",
        last_stable_checkpoint_ref="artifacts/run-5/stable.ckpt",
        certified_stable_checkpoint_ref="artifacts/run-5/cert.ckpt",
    )
    store.set_current(parent.to_dict())

    branch = create_branch_state(parent.to_dict(), "lineage-main-branch-6", "early_pretraining", parent.best_checkpoint_ref, "2026-01-02T00:00:00+00:00")
    store.set_current(branch.child_state)
    store.add_child("lineage-main", branch.child_lineage_id)

    loaded_parent = store.get_current("lineage-main")
    loaded_child = store.get_current("lineage-main-branch-6")

    assert loaded_parent is not None
    assert loaded_child is not None
    assert "lineage-main-branch-6" in loaded_parent["child_lineage_ids"]
    assert loaded_child["parent_lineage_id"] == "lineage-main"
    assert loaded_child["branch_origin_checkpoint_ref"] == "artifacts/run-5/best.ckpt"
    assert loaded_child["certified_stable_checkpoint_ref"] is None


def test_restart_records_major_intervention_and_preserves_pathologies(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    prior = LineageState(
        lineage_id="lineage-main",
        parent_lineage_id=None,
        stage_name="early_pretraining",
        status="suspect",
        trust_level="low",
        latest_run_id="run-8",
        known_pathologies=["variance_spike"],
        major_interventions=[{"type": "rollback", "run_id": "run-7", "target_checkpoint_ref": "artifacts/run-7/stable.ckpt"}],
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-08T00:00:00+00:00",
    )
    restart = create_restart_state(prior.to_dict(), "lineage-main", "early_pretraining", "2026-01-09T00:00:00+00:00", "suspect_lineage")
    store.set_current(restart.reset_state)

    loaded = store.get_current("lineage-main")

    assert loaded is not None
    assert loaded["status"] == "suspect"
    assert "variance_spike" in loaded["known_pathologies"]
    assert "suspect_lineage" in loaded["known_pathologies"]
    assert any(item.get("type") == "restart" for item in loaded["major_interventions"])


def test_orchestrator_persists_first_class_lineage_fields(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="lineage-smoke-1")
    orch.run("lineage-smoke-1")

    lineage = LineageStore(tmp_path).get_current("lineage-main")

    assert lineage is not None
    for key in ["latest_run_id", "run_count", "loop_index", "last_decision", "trust_level", "origin_run_id", "created_at", "updated_at", "metadata"]:
        assert key in lineage
    assert lineage["latest_run_id"] == "lineage-smoke-1"
