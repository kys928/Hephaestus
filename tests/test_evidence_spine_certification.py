from __future__ import annotations

from pathlib import Path

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.state.artifact_index import ArtifactIndex
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.memory_store import MemoryStore
from hephaestus.state.query import Query
from hephaestus.state.report_store import ReportStore
from hephaestus.state.run_store import RunStore


def test_evidence_spine_certification_dry_run_loop(tmp_path: Path) -> None:
    run_id = "run-evidence-spine"
    orch = build_orchestrator(state_root=tmp_path, run_id=run_id)
    results = orch.run(run_id)

    assert [result.phase for result in results] == list(SPINE_ORDER)

    run_store = RunStore(tmp_path)
    lineage_store = LineageStore(tmp_path)
    decision_store = DecisionStore(tmp_path)
    manifest_store = ManifestStore(tmp_path)
    report_store = ReportStore(tmp_path)
    memory_store = MemoryStore(tmp_path)
    artifact_index = ArtifactIndex(tmp_path)
    query = Query(tmp_path)

    run_record = run_store.get(run_id)
    assert run_record is not None

    for required in (
        "run_id",
        "lineage_id",
        "stage_name",
        "status",
        "phase_order",
        "eval_report_id",
        "judge_action",
        "checkpoint_ref",
        "data_manifest_id",
        "replay_metadata",
    ):
        assert required in run_record

    replay_metadata = dict(run_record["replay_metadata"])
    assert replay_metadata.get("replay_scope")
    assert isinstance(replay_metadata.get("requires_content_hash_match"), bool)
    if replay_metadata.get("requires_content_hash_match"):
        assert replay_metadata.get("content_hash_available") is True
        assert replay_metadata.get("checkpoint_content_hash")

    manifest_id = str(run_record["data_manifest_id"])
    manifest = manifest_store.get(manifest_id)
    assert manifest is not None
    assert manifest["manifest_id"] == manifest_id
    assert manifest["run_id"] == run_id
    assert manifest["lineage_id"] == run_record["lineage_id"]
    assert manifest.get("stage_name")
    assert manifest.get("manifest_integrity_level")
    assert "completeness_score" in manifest
    assert "missing_fields" in manifest
    assert "warnings" in manifest

    reports = report_store.all()
    eval_reports = [r for r in reports if r.get("kind") == "eval_report" and r.get("run_id") == run_id]
    assert eval_reports
    eval_report = eval_reports[-1]
    assert eval_report["kind"] == "eval_report"
    assert eval_report.get("eval_pack_id") or eval_report.get("deterministic_scorecard", {}).get("metadata", {}).get("dry_run_limited") is True
    assert eval_report.get("deterministic_scorecard")
    assert isinstance(eval_report.get("deterministic_passed"), bool)
    assert "failed_gates" in eval_report
    assert "passed_gates" in eval_report
    assert eval_report.get("scorecard_integrity_level")

    decision_id = f"dec-{run_id}-exit"
    judge_exit = decision_store.get(decision_id)
    assert judge_exit is not None
    assert judge_exit.get("decision_id") == decision_id
    metadata = dict(judge_exit.get("metadata", {}))
    for field in (
        "promotion_gate_report",
        "action_boundary",
        "effective_action",
        "blocking_failures",
        "gate_warnings",
        "promotion_allowed",
        "confidence_ceiling",
    ):
        assert field in metadata

    lineage_state = lineage_store.get_current(run_record["lineage_id"])
    assert lineage_state is not None
    assert lineage_state.get("latest_run_id") == run_id
    assert lineage_state.get("last_decision_id") == decision_id
    assert lineage_state.get("last_requested_action")
    assert lineage_state.get("last_effective_action")
    assert lineage_state.get("trust_level")
    assert lineage_state.get("status")
    assert int(lineage_state.get("run_count", 0)) >= 1
    assert int(lineage_state.get("loop_index", 0)) >= 1

    memories_for_run = memory_store.list_for_run(run_id)
    assert isinstance(memories_for_run, list)
    assert not any(
        report.get("kind") == "warning"
        and report.get("run_id") == run_id
        and report.get("warning_type") == "memory_builder_failed"
        for report in reports
    )

    lineage_id = str(run_record["lineage_id"])
    assert isinstance(query.memories_for_lineage(lineage_id), list)
    assert isinstance(query.prior_promotion_blocks(lineage_id=lineage_id), list)
    assert isinstance(query.data_issues_for_lineage(lineage_id), list)
    assert isinstance(query.eval_issues_for_lineage(lineage_id), list)
    assert isinstance(query.suspect_or_poisoned_lineages(), list)

    if hasattr(artifact_index, "all"):
        assert isinstance(artifact_index.all(), list)
