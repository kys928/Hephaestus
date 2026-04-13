from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.control.orchestrator import build_orchestrator


def test_backend_prepare_roundtrip() -> None:
    backend = LocalProcessBackend()
    prepared = backend.prepare_training_job(
        experiment_plan={"plan_id": "p1"},
        data_contract={"processed_dataset_ref": "artifacts/run-x/processed_dataset.jsonl"},
        training_plan={"run_id": "run-x", "max_steps": 200},
        launch_config={
            "artifact_root": "artifacts/run-x",
            "parameters": {"trainer_script": "tests/fixtures/fake_trainer.py"},
        },
    )
    assert prepared.run_id == "run-x"
    assert "fake_trainer.py" in " ".join(prepared.execution_spec["command"])


def test_orchestrator_real_backend_success(tmp_path: Path) -> None:
    run_id = "stage3-success"
    backend = LocalProcessBackend()
    orch = build_orchestrator(state_root=tmp_path, run_id=run_id, backend=backend)
    results = orch.run(run_id)
    outputs = {result.phase.value: result.output for result in results}

    assert outputs["runtime_monitor"]["outcome"] == "healthy"
    assert outputs["evaluator"]["checkpoint_resolution"]["selected_checkpoint_ref"].endswith(".ckpt")
    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"


def test_orchestrator_real_backend_failure_incident(tmp_path: Path) -> None:
    class FailingBackend(LocalProcessBackend):
        def prepare_training_job(self, *, experiment_plan, data_contract, training_plan, launch_config):
            launch_config = dict(launch_config)
            launch_config["parameters"] = dict(launch_config.get("parameters", {}))
            launch_config["parameters"]["force_fail"] = "1"
            return super().prepare_training_job(
                experiment_plan=experiment_plan,
                data_contract=data_contract,
                training_plan=training_plan,
                launch_config=launch_config,
            )

    run_id = "stage3-failure"
    orch = build_orchestrator(state_root=tmp_path, run_id=run_id, backend=FailingBackend())
    results = orch.run(run_id)
    outputs = {result.phase.value: result.output for result in results}

    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "failed"
    assert outputs["runtime_monitor"]["outcome"] == "hard_abort"
    assert outputs["judge_exit"]["next_action"] == "abort_run"


def test_orchestrator_missing_artifact_is_failure(tmp_path: Path) -> None:
    class MissingArtifactBackend(LocalProcessBackend):
        def prepare_training_job(self, *, experiment_plan, data_contract, training_plan, launch_config):
            launch_config = dict(launch_config)
            launch_config["parameters"] = dict(launch_config.get("parameters", {}))
            launch_config["parameters"]["omit_metrics"] = "1"
            return super().prepare_training_job(
                experiment_plan=experiment_plan,
                data_contract=data_contract,
                training_plan=training_plan,
                launch_config=launch_config,
            )

    run_id = f"stage3-missing-artifact-{uuid4().hex[:8]}"
    orch = build_orchestrator(state_root=tmp_path, run_id=run_id, backend=MissingArtifactBackend())
    results = orch.run(run_id)
    outputs = {result.phase.value: result.output for result in results}

    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "failed"
    assert outputs["runtime_monitor"]["outcome"] == "hard_abort"
    assert outputs["evaluator"]["likely_issue_category"] == "metrics_artifact_missing"
    assert outputs["judge_exit"]["next_action"] == "abort_run"


def test_dry_run_backend_still_works(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="stage3-dryrun")
    results = orch.run("stage3-dryrun")
    outputs = {result.phase.value: result.output for result in results}
    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"
