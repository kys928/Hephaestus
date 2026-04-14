from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hephaestus.backends.ardor.backend import ArdorBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.config_loader import ConfigError
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.spine import SPINE_ORDER


_FIXTURE_RUNNER = "tests/fixtures/fake_ardor_runner.py"


def _config_dir_with_ardor_runner(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs-with-runner"
    shutil.copytree("configs", config_dir)
    ardor_cfg = json.loads((config_dir / "backends" / "ardor.yaml").read_text())
    ardor_cfg["local_runner_path"] = _FIXTURE_RUNNER
    (config_dir / "backends" / "ardor.yaml").write_text(json.dumps(ardor_cfg, indent=2))
    return config_dir


def _prepare_run(backend: ArdorBackend, tmp_path: Path, run_id: str, parameters: dict[str, str] | None = None):
    processed = tmp_path / f"{run_id}-processed_dataset.jsonl"
    processed.write_text('{"text":"sample"}\n')
    payload = {"ardor_runner_script": _FIXTURE_RUNNER}
    payload.update(parameters or {})
    return backend.prepare_training_job(
        experiment_plan={"run_id": run_id},
        data_contract={"processed_dataset_ref": str(processed), "schema_version": "v1", "min_tokens": 128},
        training_plan={"run_id": run_id, "max_steps": 50, "eval_every_steps": 25, "checkpoint_every_steps": 25},
        launch_config={"backend": "ardor", "dry_run": False, "artifact_root": f"artifacts/{run_id}", "parameters": payload},
    )


def test_stage5_ardor_real_lifecycle_success(tmp_path: Path) -> None:
    backend = ArdorBackend()
    prepared = _prepare_run(backend, tmp_path, "stage5-ardor-success")
    result = backend.launch_training(prepared)

    assert result.status == "completed"
    assert result.intermediate_eval["metrics_ref"].endswith("ardor_metrics.json")
    assert len(result.checkpoint_candidates) == 2


def test_stage5_ardor_launch_failure_is_honest(tmp_path: Path) -> None:
    backend = ArdorBackend()
    prepared = _prepare_run(backend, tmp_path, "stage5-ardor-launch-fail", {"ardor_fail_launch": "1"})
    result = backend.launch_training(prepared)

    assert result.status == "failed"
    assert any("ardor_launch_failure" in event.message for event in result.events)


def test_stage5_ardor_missing_runner_path_fails_explicitly(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs-no-runner"
    shutil.copytree("configs", config_dir)
    backend = ArdorBackend(config_dir=config_dir)

    processed = tmp_path / "missing-runner-processed.jsonl"
    processed.write_text('{"text":"sample"}\n')

    with pytest.raises(ConfigError, match="requires explicit local_runner_path"):
        backend.prepare_training_job(
            experiment_plan={"run_id": "stage5-ardor-no-runner"},
            data_contract={"processed_dataset_ref": str(processed), "schema_version": "v1", "min_tokens": 128},
            training_plan={"run_id": "stage5-ardor-no-runner", "max_steps": 50, "eval_every_steps": 25, "checkpoint_every_steps": 25},
            launch_config={"backend": "ardor", "dry_run": False, "artifact_root": "artifacts/stage5-ardor-no-runner", "parameters": {}},
        )


def test_stage5_ardor_missing_metrics_fails_honestly(tmp_path: Path) -> None:
    backend = ArdorBackend()
    prepared = _prepare_run(backend, tmp_path, "stage5-ardor-missing-metrics", {"ardor_omit_metrics": "1"})
    result = backend.launch_training(prepared)

    assert result.status == "failed"
    assert any("ardor_missing_metrics_ref" in event.message for event in result.events)


def test_stage5_ardor_missing_checkpoints_fails_honestly(tmp_path: Path) -> None:
    backend = ArdorBackend()
    prepared = _prepare_run(backend, tmp_path, "stage5-ardor-missing-ckpt", {"ardor_omit_checkpoint": "1"})
    result = backend.launch_training(prepared)

    assert result.status == "failed"
    assert any("ardor_missing_checkpoint_refs" in event.message for event in result.events)


def test_stage5_ardor_malformed_contract_fails_honestly(tmp_path: Path) -> None:
    backend = ArdorBackend()
    prepared = _prepare_run(backend, tmp_path, "stage5-ardor-malformed", {"ardor_malformed_contract": "1"})
    result = backend.launch_training(prepared)

    assert result.status == "failed"
    assert any("ardor_malformed_output_contract" in event.message for event in result.events)


def test_stage5_ardor_unsupported_runtime_state_is_incident(tmp_path: Path) -> None:
    backend = ArdorBackend()
    prepared = _prepare_run(backend, tmp_path, "stage5-ardor-unsupported", {"ardor_unsupported_state": "1"})
    result = backend.launch_training(prepared)

    assert result.status == "failed"
    assert any("ardor_runtime_state_unsupported" in event.message for event in result.events)


def test_stage5_evaluator_consumes_ardor_normalized_outputs(tmp_path: Path) -> None:
    backend = ArdorBackend(config_dir=_config_dir_with_ardor_runner(tmp_path))
    orch = build_orchestrator(state_root=tmp_path, run_id="stage5-ardor-orch", backend=backend)

    artifact_root = Path("artifacts/stage5-ardor-orch")
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')

    results = orch.run("stage5-ardor-orch")
    outputs = {result.phase.value: result.output for result in results}

    assert [result.phase for result in results] == list(SPINE_ORDER)
    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"
    assert outputs["evaluator"]["checkpoint_resolution"]["selected_checkpoint_ref"].endswith(".ckpt")


def test_stage5_existing_backends_still_work(tmp_path: Path) -> None:
    dry = build_orchestrator(state_root=tmp_path / "dry", run_id="stage5-dry", backend=DryRunBackend())
    local = build_orchestrator(state_root=tmp_path / "local", run_id="stage5-local", backend=LocalProcessBackend())

    dry_outputs = {result.phase.value: result.output for result in dry.run("stage5-dry")}
    local_outputs = {result.phase.value: result.output for result in local.run("stage5-local")}

    assert dry_outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"
    assert local_outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"
