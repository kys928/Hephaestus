"""Minimal container/process entry point for infrastructure health checks."""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import threading
from collections.abc import Callable
from pathlib import Path

from .config import InfrastructureConfig
from .health import DependencyHealthService, HealthService
from .observability import JsonLineEventSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hephaestus-infrastructure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("health", help="check local liveness and readiness")
    health.add_argument("--json", action="store_true", help="emit compact JSON")
    migrate = subparsers.add_parser("migrate", help="initialize durable SQLite schemas")
    migrate.add_argument("--json", action="store_true", help="emit compact JSON")
    worker = subparsers.add_parser("worker", help="run a durable SQLite worker")
    worker.add_argument(
        "--handler",
        required=True,
        help="deployment-controlled Python handler as module:function",
    )
    worker.add_argument("--maximum-idle-polls", type=int)
    service = subparsers.add_parser(
        "service", help="run a dependency-readiness service loop without mutation APIs"
    )
    service.add_argument("--interval-seconds", type=float, default=30.0)
    service.add_argument("--once", action="store_true")
    return parser


def _root_probe(path: Path) -> str:
    return HealthService._check_root(path)


def _dependency_health(config: InfrastructureConfig, sink: JsonLineEventSink) -> DependencyHealthService:
    probes: dict[str, Callable[[], bool | str]] = {
        "state_root": lambda: _root_probe(config.state_root),
        "artifact_root": lambda: _root_probe(config.artifact_root),
        "artifact_store": lambda: _root_probe(config.artifact_root)
        if config.artifact_backend == "filesystem"
        else "capability_not_initialized",
        "secret_provider": lambda: True
        if config.secrets_backend == "environment"
        else config.secrets_root.is_dir(),
    }
    if config.queue_backend == "sqlite" or config.database_backend == "sqlite":
        from hephaestus.jobs.sqlite import SQLITE_QUEUE_SCHEMA_VERSION, SQLiteJobQueue
        from hephaestus.storage.sqlite import (
            SQLITE_LOCK_SCHEMA_VERSION,
            SQLITE_STATE_SCHEMA_VERSION,
            SQLiteLockProvider,
            SQLiteStateRepository,
        )

        queue = SQLiteJobQueue(config.sqlite_path, initialize=False, event_sink=sink)
        state = SQLiteStateRepository(config.sqlite_path, initialize=False, event_sink=sink)
        locks = SQLiteLockProvider(config.sqlite_path, initialize=False, event_sink=sink)
        probes.update(
            {
                "database": lambda: queue.ready() and state.ready(),
                "queue": queue.ready,
                "lock_service": locks.ready,
                "worker": lambda: bool(config.worker_id) and queue.ready(),
                "migrations": lambda: (
                    queue.schema_version() == SQLITE_QUEUE_SCHEMA_VERSION
                    and state.schema_version() == SQLITE_STATE_SCHEMA_VERSION
                    and locks.schema_version() == SQLITE_LOCK_SCHEMA_VERSION
                ),
            }
        )
    return DependencyHealthService(probes, config.health_required, sink)


def _migrate(config: InfrastructureConfig) -> dict[str, object]:
    if config.database_backend != "sqlite" and config.queue_backend != "sqlite":
        return {
            "migrated": False,
            "backend": config.database_backend,
            "reason": "external_migration_required",
        }
    from hephaestus.jobs.sqlite import SQLiteJobQueue
    from hephaestus.storage.sqlite import SQLiteLockProvider, SQLiteStateRepository

    queue = SQLiteJobQueue(
        config.sqlite_path,
        maximum_attempts=config.maximum_attempts,
        maximum_lease_expirations=config.maximum_lease_expirations,
    )
    state = SQLiteStateRepository(config.sqlite_path)
    locks = SQLiteLockProvider(config.sqlite_path)
    return {
        "migrated": True,
        "backend": "sqlite",
        "path": str(config.sqlite_path),
        "versions": {
            "queue": queue.schema_version(),
            "state": state.schema_version(),
            "locks": locks.schema_version(),
        },
    }


def _load_handler(reference: str):
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name or not function_name.isidentifier():
        raise ValueError("handler must use module:function syntax")
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    if not callable(handler):
        raise TypeError("configured handler is not callable")
    return handler


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = InfrastructureConfig.from_env()
    if args.command == "health":
        sink = JsonLineEventSink(config.event_log_path)
        report = _dependency_health(config, sink).check()
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.ready else 1
    if args.command == "migrate":
        payload = _migrate(config)
        print(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":") if args.json else None,
                indent=None if args.json else 2,
            )
        )
        return 0 if payload["migrated"] else 1
    if args.command == "worker":
        if config.queue_backend != "sqlite":
            raise RuntimeError("the built-in durable worker command requires the SQLite queue")
        from hephaestus.jobs import DurableWorker, SQLiteJobQueue

        queue = SQLiteJobQueue(
            config.sqlite_path,
            maximum_attempts=config.maximum_attempts,
            maximum_lease_expirations=config.maximum_lease_expirations,
            initialize=False,
        )
        if not queue.ready():
            raise RuntimeError("durable queue schema is not ready; run migrate first")
        worker = DurableWorker(
            config.worker_id,
            queue,
            _load_handler(args.handler),
            lease_seconds=config.lease_seconds,
            poll_interval_seconds=config.queue_poll_seconds,
            event_sink=JsonLineEventSink(config.event_log_path),
        )
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_number, lambda *_: worker.request_shutdown())
        worker.run_forever(maximum_idle_polls=args.maximum_idle_polls)
        return 0
    if args.command == "service":
        if args.interval_seconds <= 0:
            raise ValueError("interval-seconds must be positive")
        stopped = threading.Event()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_number, lambda *_: stopped.set())
        sink = JsonLineEventSink(config.event_log_path)
        while True:
            report = _dependency_health(config, sink).check()
            print(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")), flush=True)
            if args.once:
                return 0 if report.ready else 1
            if stopped.wait(args.interval_seconds):
                return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
