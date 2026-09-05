"""Generic Hephaestus product CLI."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionLoopRunner


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("program config must be a JSON object")
    return payload


def _driver(reference: str, config: dict[str, object]):
    if ":" not in reference:
        raise ValueError("driver must be an installed module:function reference")
    module_name, function_name = reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory(config)


def _composition(payload: dict[str, Any]) -> ProductionCompositionRoot:
    raw = payload.get("composition", {})
    if not isinstance(raw, dict):
        raise ValueError("composition must be an object")
    state_root = Path(str(raw.get("state_root", "state/production")))
    artifact_root = Path(str(raw.get("artifact_root", "artifacts/production")))
    settings = ProductionCompositionSettings(
        state_root=state_root,
        artifact_root=artifact_root,
        cache_root=Path(str(raw["cache_root"])) if raw.get("cache_root") else None,
        database_path=Path(str(raw["database_path"])) if raw.get("database_path") else None,
        model_catalog_path=Path(str(raw["model_catalog_path"])) if raw.get("model_catalog_path") else None,
        enable_dataset_network=bool(raw.get("enable_dataset_network", False)),
        dataset_provider_allowlist=tuple(str(item) for item in raw.get("dataset_provider_allowlist", ["huggingface"])),
        maximum_training_steps=int(raw.get("maximum_training_steps", 100_000)),
        maximum_dataset_bytes=int(raw.get("maximum_dataset_bytes", 512 * 1024 * 1024)),
        maximum_dataset_rows=int(raw.get("maximum_dataset_rows", 1_000_000)),
        maximum_infrastructure_attempts=int(raw.get("maximum_infrastructure_attempts", 5)),
    )
    return ProductionCompositionRoot(settings)


def run_program(path: str | Path, *, resume: bool = True) -> dict[str, object]:
    payload = _load_json(path)
    root = _composition(payload)
    runtime = root.build()
    driver_ref = str(payload.get("driver", "")).strip()
    if not driver_ref:
        raise ValueError("program config requires driver=module:function")
    raw_driver_config = payload.get("driver_config", {})
    driver_config = dict(raw_driver_config) if isinstance(raw_driver_config, dict) else {}
    driver = _driver(driver_ref, driver_config)
    runner = ProductionLoopRunner(
        runtime,
        driver,
        maximum_cycles=int(payload.get("maximum_cycles", 8)),
        stop_on_promotion=bool(payload.get("stop_on_promotion", True)),
    )
    state = runner.run(
        program_id=str(payload["program_id"]),
        lineage_id=str(payload["lineage_id"]),
        stage_name=str(payload.get("stage_name", "smoke_test")),
        resume=resume,
    )
    return {
        "program": state.to_dict(),
        "services": runtime.service_inventory(),
        "database": str(runtime.state_repository.path),
        "artifact_root": str(runtime.settings.artifact_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hephaestus", description="Run the governed Hephaestus production loop.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start or resume a governed multi-experiment program")
    run.add_argument("--config", required=True)
    run.add_argument("--fresh", action="store_true", help="refuse previously persisted loop state")
    inspect = sub.add_parser("status", help="read persisted production-loop state")
    inspect.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_program(args.config, resume=not args.fresh)
        print(json.dumps(result, indent=2, sort_keys=True))
        status = str(result["program"].get("status", ""))  # type: ignore[union-attr]
        return 0 if status in {"completed", "stopped"} else 2

    payload = _load_json(args.config)
    runtime = _composition(payload).build()
    state = runtime.loop_state.get(str(payload["program_id"]))
    print(json.dumps({"program": None if state is None else state.to_dict(), "services": runtime.service_inventory()}, indent=2, sort_keys=True))
    return 0 if state is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
