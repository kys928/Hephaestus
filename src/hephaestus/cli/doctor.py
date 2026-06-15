from __future__ import annotations

import argparse
import json
from pathlib import Path

from hephaestus.schemas.demo_cli import DoctorPayload


def check_state_root(state_root: Path) -> dict[str, object]:
    exists = state_root.exists()
    return DoctorPayload(
        state_root=str(state_root),
        exists=exists,
        is_dir=state_root.is_dir() if exists else False,
        created=False,
        status="ok" if exists and state_root.is_dir() else "missing",
        checks=[
            {"name": "state_root_exists", "ok": exists},
            {"name": "state_root_is_directory", "ok": state_root.is_dir() if exists else False},
        ],
    ).to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only local Hephaestus environment doctor.")
    parser.add_argument("--state-root", default="state", help="Existing state root to inspect; never created.")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)
    payload = check_state_root(Path(args.state_root))
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"HEPHAESTUS DOCTOR\nstate_root: {payload['state_root']}\nstatus: {payload['status']}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
