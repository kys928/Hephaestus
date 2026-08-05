"""Minimal container/process entry point for infrastructure health checks."""

from __future__ import annotations

import argparse
import json

from .config import InfrastructureConfig
from .health import HealthService
from .observability import JsonLineEventSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hephaestus-infrastructure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("health", help="check local liveness and readiness")
    health.add_argument("--json", action="store_true", help="emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = InfrastructureConfig.from_env()
    if args.command == "health":
        report = HealthService(
            config.state_root,
            config.artifact_root,
            JsonLineEventSink(config.event_log_path),
        ).check()
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.ready else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
