from __future__ import annotations

import argparse
import json
from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.replay_verification import verify_run_replay
from hephaestus.state.run_store import RunStore


def _next_default_run_id(state_root: Path) -> str:
    store = RunStore(state_root)
    if store.get("demo-run") is None:
        return "demo-run"
    index = 2
    while store.get(f"demo-run-{index}") is not None:
        index += 1
    return f"demo-run-{index}"


def create_demo_state(
    state_root: Path,
    run_id: str | None = None,
    lineage_id: str = "lineage-demo",
    stage_name: str = "early_pretraining",
) -> dict[str, object]:
    state_root.mkdir(parents=True, exist_ok=True)
    selected_run_id = run_id or _next_default_run_id(state_root)
    orchestrator = build_orchestrator(
        state_root=state_root,
        run_id=selected_run_id,
        lineage_id=lineage_id,
        stage_name=stage_name,
        backend=DryRunBackend(),
    )
    phase_results = orchestrator.run(selected_run_id)
    run = RunStore(state_root).get(selected_run_id) or {}
    replay = verify_run_replay(state_root, selected_run_id).to_dict()
    return {
        "state_root": str(state_root),
        "run_id": selected_run_id,
        "lineage_id": lineage_id,
        "stage_name": stage_name,
        "run_status": run.get("status"),
        "judge_action": run.get("judge_action"),
        "checkpoint_ref": run.get("checkpoint_ref"),
        "phase_count": len(phase_results),
        "replay_status": replay.get("status"),
        "dry_run_note": "Dry-run artifact refs are references only; no real model artifact files or training outputs are created.",
        "next_commands": {
            "inspect_run": f"python -m hephaestus.cli.inspect_run --state-root {state_root} --run-id {selected_run_id} --format json",
            "verify_replay": f"python -m hephaestus.cli.verify_replay --state-root {state_root} --run-id {selected_run_id} --format json",
            "operator_console": f"python -m hephaestus.app.console --state-root {state_root}",
        },
    }


def _text(payload: dict[str, object]) -> str:
    commands = payload.get("next_commands") or {}
    lines = [
        "HEPHAESTUS DEMO STATE",
        f"state_root: {payload.get('state_root')}",
        f"run_id: {payload.get('run_id')}",
        f"run_status: {payload.get('run_status')}",
        f"replay_status: {payload.get('replay_status')}",
        f"dry_run_note: {payload.get('dry_run_note')}",
        "next_commands:",
    ]
    if isinstance(commands, dict):
        lines.extend(f"- {name}: {command}" for name, command in commands.items())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic local demo Hephaestus state using the dry-run orchestrator.")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lineage-id", default="lineage-demo")
    parser.add_argument("--stage-name", default="early_pretraining")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)
    payload = create_demo_state(Path(args.state_root), args.run_id, args.lineage_id, args.stage_name)
    print(json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else _text(payload))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
