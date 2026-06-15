from __future__ import annotations

import argparse
import json
from pathlib import Path

from hephaestus.schemas.demo_cli import DemoStatePayload
from hephaestus.state.artifact_index import ArtifactIndex
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.memory_store import MemoryStore
from hephaestus.state.report_store import ReportStore
from hephaestus.state.run_store import RunStore

_TS = "2026-01-01T00:00:00Z"


def create_demo_state(state_root: Path, run_id: str = "demo-run") -> dict[str, object]:
    state_root.mkdir(parents=True, exist_ok=True)
    lineage_id = f"lineage-{run_id}"
    manifest_id = f"manifest-{run_id}"
    eval_id = f"eval-{run_id}"
    checkpoint_ref = f"checkpoints/{run_id}/candidate"

    LineageStore(state_root).set_current({
        "lineage_id": lineage_id,
        "stage_name": "smoke_test",
        "status": "exploratory",
        "trust_level": "demo",
        "origin_run_id": run_id,
        "created_at": _TS,
        "updated_at": _TS,
        "latest_run_id": run_id,
        "best_checkpoint_ref": checkpoint_ref,
        "last_decision": "hold",
        "last_decision_id": f"dec-{run_id}-exit",
        "run_count": 1,
    })
    RunStore(state_root).append({
        "run_id": run_id,
        "lineage_id": lineage_id,
        "stage_name": "smoke_test",
        "status": "completed",
        "artifact_root": f"artifacts/{run_id}",
        "started_at": _TS,
        "completed_at": _TS,
        "phase_order": ["judge_entry", "planner", "data_acquisition_audit", "data_preprocessor", "training_engineer", "runtime_monitor", "evaluator", "judge_exit"],
        "monitor_outcome": "demo_only_no_training_launch",
        "eval_report_id": eval_id,
        "judge_action": "hold",
        "loop_index": 0,
        "checkpoint_ref": checkpoint_ref,
        "data_manifest_id": manifest_id,
        "replay_metadata": {"replay_scope": "decision_context", "checkpoint_ref": checkpoint_ref, "checkpoint_content_hash": "demo-hash", "content_hash_available": True, "requires_content_hash_match": True},
    })
    ManifestStore(state_root).append({
        "manifest_id": manifest_id,
        "run_id": run_id,
        "lineage_id": lineage_id,
        "artifact_ref": f"data/{run_id}/manifest.json",
        "source_refs": ["demo://synthetic-read-only"],
        "record_count": 1,
        "content_hash": "demo-data-hash",
    })
    ReportStore(state_root).append({
        "kind": "eval_report",
        "eval_id": eval_id,
        "run_id": run_id,
        "lineage_id": lineage_id,
        "status": "completed",
        "checkpoint_resolution": {"selected_checkpoint_ref": checkpoint_ref},
        "deterministic_scorecard": {"evidence_refs": [f"reports/{run_id}/scorecard.json"]},
        "intermediate_artifact_refs": [f"reports/{run_id}/eval.json"],
        "replay_metadata": {"replay_scope": "decision_context", "checkpoint_ref": checkpoint_ref},
    })
    DecisionStore(state_root).append({
        "decision_id": f"dec-{run_id}-exit",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "role": "judge_exit",
        "action": "hold",
        "effective_action": "hold",
        "evidence_refs": [f"reports/{run_id}/eval.json"],
        "metadata": {"checkpoint_ref": checkpoint_ref, "promotion_gate_report": {"status": "not_eligible"}, "action_boundary": {"training_launch": False, "ui_mutation": False}, "replay_requirements": {"replay_scope": "decision_context"}},
    })
    MemoryStore(state_root).append({"memory_id": f"memory-{run_id}", "run_id": run_id, "lineage_id": lineage_id, "memory_type": "demo_summary", "source_kind": "demo_cli", "source_id": run_id, "summary": "Local demo state only; no training was launched.", "evidence_refs": [f"reports/{run_id}/eval.json"], "tags": ["demo", "read_only"]})
    ArtifactIndex(state_root).append({"run_id": run_id, "ref": f"reports/{run_id}/eval.json", "kind": "demo_report"})
    return DemoStatePayload(state_root=str(state_root), run_id=run_id, lineage_id=lineage_id, created=True).to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic local demo Hephaestus state.")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--run-id", default="demo-run")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)
    payload = create_demo_state(Path(args.state_root), args.run_id)
    print(json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else f"created demo run {args.run_id} at {args.state_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
