from __future__ import annotations

import argparse
import json
from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.replay_verification import verify_run_replay
from hephaestus.state.run_store import RunStore

DEFAULT_RUN_ID = "demo-run"
DEFAULT_LINEAGE_ID = "lineage-demo"
DEFAULT_STAGE_NAME = "early_pretraining"
DRY_RUN_NOTE = "Dry-run artifact refs are references only; no real model artifact files or training outputs are created."


def _next_default_run_id(state_root: Path) -> str:
    existing = {str(row.get("run_id")) for row in RunStore(state_root).all()}
    if DEFAULT_RUN_ID not in existing:
        return DEFAULT_RUN_ID
    index = 2
    while f"{DEFAULT_RUN_ID}-{index}" in existing:
        index += 1
    return f"{DEFAULT_RUN_ID}-{index}"


def _commands(state_root: Path, run_id: str) -> dict[str, str]:
    root = str(state_root)
    return {
        "inspect_run": f"python -m hephaestus.cli.inspect_run --state-root {root} --run-id {run_id} --format text",
        "verify_replay": f"python -m hephaestus.cli.verify_replay --state-root {root} --run-id {run_id} --format text",
        "operator_console": f"python -m hephaestus.app.console --state-root {root} --host 127.0.0.1 --port 8765",
    }


def create_demo_state(
    state_root: Path,
    run_id: str | None = None,
    lineage_id: str = DEFAULT_LINEAGE_ID,
    stage_name: str = DEFAULT_STAGE_NAME,
) -> dict[str, object]:
    chosen_run_id = run_id or _next_default_run_id(state_root)
    if run_id and RunStore(state_root).get(run_id):
        raise ValueError(f"run_id '{run_id}' already exists; choose a new --run-id")

    orchestrator = build_orchestrator(
        state_root=state_root,
        run_id=chosen_run_id,
        lineage_id=lineage_id,
        stage_name=stage_name,
        backend=DryRunBackend(),
    )
    phase_results = orchestrator.run(chosen_run_id)
    run_record = RunStore(state_root).get(chosen_run_id) or {}
    replay = verify_run_replay(state_root, chosen_run_id).to_dict()
    return {
        "state_root": str(state_root),
        "run_id": chosen_run_id,
        "lineage_id": lineage_id,
        "stage_name": stage_name,
        "run_status": run_record.get("status"),
        "judge_action": run_record.get("judge_action"),
        "checkpoint_ref": run_record.get("checkpoint_ref"),
        "phase_count": len(phase_results),
        "replay_status": replay.get("status"),
        "dry_run_note": DRY_RUN_NOTE,
        "next_commands": _commands(state_root, chosen_run_id),
    }


def _text(payload: dict[str, object]) -> str:
    commands = dict(payload.get("next_commands") or {})
    return "\n".join(
        [
            "HEPHAESTUS DEMO STATE CREATED",
            f"state_root: {payload.get('state_root')}",
            f"run_id: {payload.get('run_id')}",
            f"run_status: {payload.get('run_status')}",
            f"replay_status: {payload.get('replay_status')}",
            f"note: {payload.get('dry_run_note')}",
            "next_commands:",
            *(f"- {name}: {command}" for name, command in commands.items()),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a local Hephaestus dry-run demo state.")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--lineage-id", default=DEFAULT_LINEAGE_ID)
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    args = parser.parse_args(argv)

    try:
        payload = create_demo_state(
            state_root=Path(args.state_root),
            run_id=args.run_id,
            lineage_id=args.lineage_id,
            stage_name=args.stage_name,
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
