from __future__ import annotations

import argparse
import json
from pathlib import Path

from hephaestus.control.replay_verification import verify_run_replay


def _text(payload: dict[str, object]) -> str:
    missing = payload.get("missing_evidence") or []
    warnings = payload.get("warnings") or []
    evidence = payload.get("evidence_refs") or []
    return "\n".join(
        [
            "HEPHAESTUS REPLAY VERIFICATION",
            f"run_id: {payload.get('run_id')}",
            f"lineage_id: {payload.get('lineage_id')}",
            f"status: {payload.get('status')}",
            f"replay_scope: {payload.get('replay_scope')}",
            f"checkpoint_ref: {payload.get('checkpoint_ref')}",
            f"content_hash_available: {payload.get('content_hash_available')}",
            f"requires_content_hash_match: {payload.get('requires_content_hash_match')}",
            "missing_evidence:",
            *(f"- {item}" for item in missing),
            "warnings:",
            *(f"- {item}" for item in warnings),
            "evidence_refs:",
            *(f"- {item}" for item in evidence),
            f"summary: {payload.get('summary')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify read-only replay evidence for a Hephaestus run.")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)

    report = verify_run_replay(Path(args.state_root), args.run_id)
    payload = report.to_dict()
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
