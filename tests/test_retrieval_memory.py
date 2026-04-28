from __future__ import annotations

from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.memory_builder import build_memory_records_from_run
from hephaestus.state.memory_store import MemoryStore
from hephaestus.state.query import Query
from hephaestus.state.report_store import ReportStore


def test_memory_store_round_trip_and_source_linkage(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    record = {
        "memory_id": "mem-1",
        "memory_type": "promotion_block",
        "source_kind": "decision",
        "source_id": "dec-1",
        "lineage_id": "lineage-main",
        "run_id": "run-1",
        "stage_name": "early_pretraining",
        "created_at": "2026-01-01T00:00:00+00:00",
        "severity": "major",
        "summary": "blocked by deterministic gate",
        "tags": ["promotion_block", "deterministic_scorecard"],
        "evidence_refs": ["artifacts/run-1/eval.json"],
        "related_ids": ["eval-run-1"],
        "confidence": 0.95,
        "metadata": {"gate": "deterministic_scorecard"},
    }

    store.append(record)

    assert store.get("mem-1") is not None
    assert store.list_for_run("run-1")[0]["source_id"] == "dec-1"
    assert store.list_for_lineage("lineage-main")[0]["source_kind"] == "decision"
    assert store.list_all()[0]["memory_type"] == "promotion_block"


def test_promotion_block_extraction_from_decision_metadata() -> None:
    records = build_memory_records_from_run(
        run_id="run-1",
        run_record={"run_id": "run-1", "lineage_id": "lineage-main", "stage_name": "early_pretraining"},
        lineage_state={"lineage_id": "lineage-main", "status": "active", "recent_failures": []},
        decisions=[
            {
                "decision_id": "dec-run-1-exit",
                "run_id": "run-1",
                "metadata": {
                    "requested_action": "promote_checkpoint",
                    "promotion_gate_report": {
                        "blocking_failures": ["Deterministic evidence missing"],
                        "gates": [{"gate_id": "deterministic_scorecard", "blocking": True, "severity": "critical"}],
                    },
                },
            }
        ],
        reports=[],
        manifests=[],
    )

    blocks = [row for row in records if row.memory_type == "promotion_block"]
    assert len(blocks) == 1
    assert blocks[0].severity == "critical"
    assert blocks[0].source_id == "dec-run-1-exit"


def test_data_and_eval_issue_extraction() -> None:
    records = build_memory_records_from_run(
        run_id="run-2",
        run_record={"run_id": "run-2", "lineage_id": "lineage-main", "stage_name": "early_pretraining"},
        lineage_state={"lineage_id": "lineage-main", "status": "active", "recent_failures": []},
        decisions=[],
        reports=[
            {
                "kind": "eval_report",
                "run_id": "run-2",
                "eval_id": "eval-run-2",
                "eval_pack_integrity_level": "reference_only",
                "scorecard_integrity_level": "insufficient",
                "deterministic_scorecard": {},
            }
        ],
        manifests=[
            {
                "manifest_id": "m-run-2",
                "run_id": "run-2",
                "manifest_integrity_level": "insufficient",
                "warnings": ["missing_source_hash"],
                "artifact_ref": "artifacts/run-2/manifest.json",
            }
        ],
    )

    assert any(row.memory_type == "data_issue" for row in records)
    assert any(row.memory_type == "eval_issue" for row in records)


def test_known_dead_end_extraction_for_poisoned_lineage() -> None:
    records = build_memory_records_from_run(
        run_id="run-3",
        run_record={"run_id": "run-3", "lineage_id": "lineage-main", "stage_name": "early_pretraining"},
        lineage_state={"lineage_id": "lineage-main", "status": "poisoned", "recent_failures": ["f-1"]},
        decisions=[],
        reports=[],
        manifests=[],
    )

    dead_ends = [row for row in records if row.memory_type == "known_dead_end"]
    assert len(dead_ends) == 1
    assert dead_ends[0].severity == "critical"


def test_query_methods_for_memory_and_lineage_status(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path)
    memory_store.append(
        {
            "memory_id": "mem-pb",
            "memory_type": "promotion_block",
            "source_kind": "decision",
            "source_id": "dec-9",
            "lineage_id": "lineage-main",
            "run_id": "run-9",
            "stage_name": "early_pretraining",
            "created_at": None,
            "severity": "major",
            "summary": "blocked",
            "tags": ["promotion_block"],
            "evidence_refs": [],
            "related_ids": [],
            "confidence": 0.9,
            "metadata": {},
        }
    )
    memory_store.append(
        {
            "memory_id": "mem-data",
            "memory_type": "data_issue",
            "source_kind": "manifest",
            "source_id": "m-9",
            "lineage_id": "lineage-main",
            "run_id": "run-9",
            "stage_name": "early_pretraining",
            "created_at": None,
            "severity": "warning",
            "summary": "integrity limited",
            "tags": ["data_manifest"],
            "evidence_refs": [],
            "related_ids": [],
            "confidence": 0.7,
            "metadata": {},
        }
    )

    lineage_store = LineageStore(tmp_path)
    lineage_store.set_current(
        LineageState(
            lineage_id="lineage-main",
            stage_name="early_pretraining",
            status="poisoned",
            trust_level="degraded",
            origin_run_id="run-0",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ).to_dict()
    )

    query = Query(tmp_path)
    assert len(query.prior_promotion_blocks("lineage-main")) == 1
    assert len(query.data_issues_for_lineage("lineage-main")) == 1
    assert len(query.suspect_or_poisoned_lineages()) == 1


def test_orchestrator_smoke_appends_memory_records(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(state_root=tmp_path, run_id="run-memory", backend=DryRunBackend())
    orchestrator.run("run-memory")

    memories = MemoryStore(tmp_path).list_for_run("run-memory")
    assert memories


def test_memory_builder_failures_do_not_break_run(tmp_path: Path, monkeypatch) -> None:
    from hephaestus.control import orchestrator as orchestrator_module

    def _explode(**_: object) -> list[object]:
        raise RuntimeError("memory failure")

    monkeypatch.setattr(orchestrator_module, "build_memory_records_from_run", _explode)
    orchestrator = build_orchestrator(state_root=tmp_path, run_id="run-memory-warn", backend=DryRunBackend())

    results = orchestrator.run("run-memory-warn")

    assert results
    warnings = [row for row in ReportStore(tmp_path).all() if row.get("warning_type") == "memory_builder_failed"]
    assert warnings


def test_memory_deduplication_when_built_twice(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    records = build_memory_records_from_run(
        run_id="run-dup",
        run_record={"run_id": "run-dup", "lineage_id": "lineage-main", "stage_name": "early_pretraining"},
        lineage_state={"lineage_id": "lineage-main", "status": "poisoned", "recent_failures": ["f-1", "f-1"]},
        decisions=[
            {
                "decision_id": "dec-run-dup-exit",
                "run_id": "run-dup",
                "metadata": {
                    "requested_action": "promote_checkpoint",
                    "promotion_gate_report": {
                        "blocking_failures": ["Deterministic evidence missing"],
                        "gates": [{"gate_id": "deterministic_scorecard", "blocking": True, "severity": "critical"}],
                    },
                },
            }
        ],
        reports=[],
        manifests=[],
    )

    for row in records:
        store.append(row)
    for row in records:
        store.append(row)

    assert len(store.list_for_run("run-dup")) == len(records)
