from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

IMPORT_KEYS = ("hephaestus", "build_orchestrator", "inspect_run_cli", "verify_replay_cli", "verify_run_replay")
STABLE_KEYS = (
    "status",
    "python_version",
    "imports",
    "state_root",
    "state_root_provided",
    "state_root_exists",
    "run_count",
    "latest_run_id",
    "latest_replay_status",
    "warnings",
    "errors",
)


def _check_imports() -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    specs = {
        "hephaestus": ("hephaestus", None),
        "build_orchestrator": ("hephaestus.control.orchestrator", "build_orchestrator"),
        "inspect_run_cli": ("hephaestus.cli.inspect_run", "main"),
        "verify_replay_cli": ("hephaestus.cli.verify_replay", "main"),
        "verify_run_replay": ("hephaestus.control.replay_verification", "verify_run_replay"),
    }
    for key, (module_name, attr) in specs.items():
        try:
            module = importlib.import_module(module_name)
            if attr is not None:
                getattr(module, attr)
            checks[key] = True
        except Exception as exc:  # pragma: no cover - defensive diagnostic path
            checks[key] = False
            errors.append(f"import_failed:{key}:{exc}")
    return checks, errors


def doctor(state_root: Path | None = None) -> dict[str, object]:
    imports, errors = _check_imports()
    warnings: list[str] = []
    state_root_provided = state_root is not None
    state_root_exists = state_root.exists() if state_root_provided else False
    run_count = 0
    latest_run_id: str | None = None
    latest_replay_status: str | None = None

    if state_root_provided and not state_root_exists:
        warnings.append("state_root_missing")
    elif state_root_provided and state_root_exists:
        from hephaestus.control.replay_verification import verify_run_replay
        from hephaestus.state.run_store import RunStore

        runs = RunStore(state_root).all()
        run_count = len(runs)
        if runs:
            latest = runs[-1]
            latest_run_id = str(latest.get("run_id") or "") or None
            if latest_run_id:
                latest_replay_status = str(verify_run_replay(state_root, latest_run_id).status)
                if latest_replay_status in {"insufficient", "missing"}:
                    warnings.append(f"latest_replay_{latest_replay_status}")
            else:
                warnings.append("latest_run_missing_run_id")
        else:
            warnings.append("state_root_has_no_runs")

    critical_imports_ok = all(imports.values())
    if not critical_imports_ok:
        status = "failed"
    elif state_root_provided and (not state_root_exists or run_count == 0 or latest_replay_status in {"insufficient", "missing"}):
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "python_version": sys.version.split()[0],
        "imports": {key: bool(imports.get(key)) for key in IMPORT_KEYS},
        "state_root": str(state_root) if state_root_provided else None,
        "state_root_provided": state_root_provided,
        "state_root_exists": state_root_exists,
        "run_count": run_count,
        "latest_run_id": latest_run_id,
        "latest_replay_status": latest_replay_status,
        "warnings": warnings,
        "errors": errors,
    }


def _text(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "HEPHAESTUS DOCTOR",
            f"status: {payload.get('status')}",
            f"python_version: {payload.get('python_version')}",
            f"state_root: {payload.get('state_root')}",
            f"state_root_exists: {payload.get('state_root_exists')}",
            f"run_count: {payload.get('run_count')}",
            f"latest_run_id: {payload.get('latest_run_id')}",
            f"latest_replay_status: {payload.get('latest_replay_status')}",
            "warnings:",
            *(f"- {item}" for item in payload.get("warnings", [])),
            "errors:",
            *(f"- {item}" for item in payload.get("errors", [])),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local Hephaestus operator readiness without writing state.")
    parser.add_argument("--state-root")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)
    payload = doctor(Path(args.state_root) if args.state_root else None)
    if args.format == "json":
        print(json.dumps({key: payload[key] for key in STABLE_KEYS}, indent=2, sort_keys=True))
    else:
        print(_text(payload))
    return 1 if payload["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
