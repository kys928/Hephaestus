from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

_IMPORTS = {
    "hephaestus": "hephaestus",
    "build_orchestrator": "hephaestus.control.orchestrator:build_orchestrator",
    "inspect_run_cli": "hephaestus.cli.inspect_run:main",
    "verify_replay_cli": "hephaestus.cli.verify_replay:main",
    "verify_run_replay": "hephaestus.control.replay_verification:verify_run_replay",
    "operator_console": "hephaestus.app.console:make_handler",
}


def _check_imports() -> tuple[dict[str, dict[str, object]], list[str]]:
    results: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for name, target in _IMPORTS.items():
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
            if attr:
                getattr(module, attr)
            results[name] = {"ok": True, "target": target}
        except Exception as exc:  # pragma: no cover - diagnostics path
            results[name] = {"ok": False, "target": target, "error": str(exc)}
            errors.append(f"import_failed:{name}:{exc}")
    return results, errors


def check(state_root: Path | None = None) -> dict[str, Any]:
    imports, errors = _check_imports()
    warnings: list[str] = []
    provided = state_root is not None
    exists = state_root.exists() if state_root is not None else False
    run_count = 0
    latest_run_id: str | None = None
    latest_replay_status: str | None = None

    if provided:
        if not exists:
            warnings.append("state_root_missing")
        elif not state_root.is_dir():
            errors.append("state_root_not_directory")
        else:
            from hephaestus.control.replay_verification import verify_run_replay
            from hephaestus.state.run_store import RunStore

            runs = RunStore(state_root).all()
            run_count = len(runs)
            if not runs:
                warnings.append("state_root_empty")
            else:
                latest_run_id = str(runs[-1].get("run_id") or "") or None
                if latest_run_id:
                    latest_replay_status = str(verify_run_replay(state_root, latest_run_id).to_dict().get("status") or "") or None
                    if latest_replay_status not in {"verified", "partial"}:
                        warnings.append("latest_replay_insufficient")
                else:
                    warnings.append("latest_run_id_missing")

    critical_failed = any(not result.get("ok") for result in imports.values())
    status = "failed" if critical_failed or errors else "warning" if warnings else "ok"
    return {
        "status": status,
        "python_version": sys.version.split()[0],
        "imports": imports,
        "state_root": str(state_root) if state_root is not None else None,
        "state_root_provided": provided,
        "state_root_exists": exists,
        "run_count": run_count,
        "latest_run_id": latest_run_id,
        "latest_replay_status": latest_replay_status,
        "warnings": warnings,
        "errors": errors,
    }


def _text(payload: dict[str, Any]) -> str:
    return "\n".join([
        "HEPHAESTUS DOCTOR",
        f"status: {payload['status']}",
        f"python_version: {payload['python_version']}",
        f"state_root: {payload['state_root']}",
        f"run_count: {payload['run_count']}",
        f"latest_run_id: {payload['latest_run_id']}",
        f"latest_replay_status: {payload['latest_replay_status']}",
        f"warnings: {', '.join(payload['warnings']) if payload['warnings'] else 'none'}",
        f"errors: {', '.join(payload['errors']) if payload['errors'] else 'none'}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only local Hephaestus environment doctor.")
    parser.add_argument("--state-root", default=None, help="Existing state root to inspect; never created.")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)
    payload = check(Path(args.state_root) if args.state_root else None)
    print(json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else _text(payload))
    return 1 if payload["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
