from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from hephaestus.backends.ardor.backend import ArdorBackend
from hephaestus.backends.ardor.eval_adapter import ArdorEvalAdapter
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.backends.registry import resolve_backend
from hephaestus.config_loader import ConfigError, load_config_file
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.roles.training_engineer import TrainingEngineerRole
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


def test_stage_profile_loads_from_config() -> None:
    profile = StagePolicy().resolve("continuation_repair")
    assert profile.strictness == "strict"
    assert profile.deterministic_gates["min_probe_score"] == 0.75
    assert "promote_checkpoint" in profile.allowed_next_actions


def test_eval_pack_loads_from_config() -> None:
    pack = load_eval_pack("pretraining_probes")
    assert pack["pack_name"] == "pretraining_probes"
    assert pack["required_metrics"] == ["probe_score", "toxicity"]


def test_recipe_template_affects_training_plan() -> None:
    plan, launch = TrainingEngineerRole().run(
        run_id="stage4-plan",
        stage_name="early_pretraining",
        artifact_root="artifacts/stage4-plan",
        data_contract={"processed_dataset_ref": "artifacts/stage4-plan/processed_dataset.jsonl"},
        backend_name="dry_run",
        dry_run=True,
    )
    assert plan.max_steps == 300
    assert launch.parameters["init_mode"] == "from_scratch"


def test_backend_registry_resolves_ardor() -> None:
    backend = resolve_backend("ardor")
    assert isinstance(backend, ArdorBackend)


def test_ardor_backend_prepares_job_without_orchestrator_leakage(tmp_path: Path) -> None:
    run_id = "stage4-ardor-prepare"
    processed = tmp_path / "processed_dataset.jsonl"
    processed.write_text('{"text":"sample"}\n')

    backend = ArdorBackend()
    prepared = backend.prepare_training_job(
        experiment_plan={"run_id": run_id},
        data_contract={
            "processed_dataset_ref": str(processed),
            "schema_version": "v1",
            "min_tokens": 128,
        },
        training_plan={"run_id": run_id, "max_steps": 50, "eval_every_steps": 25, "checkpoint_every_steps": 25},
        launch_config={"backend": "ardor", "dry_run": False, "artifact_root": f"artifacts/{run_id}", "parameters": {"ardor_runner_script": "tests/fixtures/fake_ardor_runner.py"}},
    )

    assert prepared.execution_spec["runner"] == "ardor_local_process"
    assert "dataset" in prepared.execution_spec["job_spec"]


def test_ardor_unsupported_model_fails_honestly(tmp_path: Path) -> None:
    backend = ArdorBackend()
    processed = tmp_path / "processed_dataset.jsonl"
    processed.write_text('{"text":"sample"}\n')

    with pytest.raises(ConfigError):
        backend.prepare_training_job(
            experiment_plan={"run_id": "bad-model"},
            data_contract={"processed_dataset_ref": str(processed), "schema_version": "v1", "min_tokens": 128},
            training_plan={"run_id": "bad-model", "max_steps": 10, "eval_every_steps": 5, "checkpoint_every_steps": 5},
            launch_config={
                "backend": "ardor",
                "artifact_root": "artifacts/bad-model",
                "parameters": {"model_id": "missing-model"},
            },
        )


def test_ardor_missing_processed_dataset_fails_honestly() -> None:
    backend = ArdorBackend()
    run_id = f"stage4-managed-{uuid4().hex[:8]}"
    with pytest.raises(ConfigError, match="processed dataset missing"):
        backend.preprocess(run_id)


def test_ardor_eval_adapter_rejects_unsupported_fields() -> None:
    adapter = ArdorEvalAdapter()
    with pytest.raises(ConfigError, match="unsupported fields"):
        adapter.adapt_intermediate_eval({"metrics_ref": "m.json", "generation_ref": "gen.json"})


def test_dry_run_backend_still_operates(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="stage4-dry", backend=DryRunBackend())
    outputs = {result.phase.value: result.output for result in orch.run("stage4-dry")}
    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"


class OneIncidentBackend(LocalProcessBackend):
    def launch_training(self, prepared_job):
        base = super().launch_training(prepared_job)
        base.events.append(
            RuntimeEvent(
                event_id=f"{prepared_job.run_id}-incident",
                run_id=prepared_job.run_id,
                step=10,
                category=RuntimeEventCategory.INCIDENT,
                message="single incident",
                payload_ref="artifacts/incident.json",
            )
        )
        return base


class TwoIncidentBackend(LocalProcessBackend):
    def launch_training(self, prepared_job):
        base = super().launch_training(prepared_job)
        base.events.append(
            RuntimeEvent(
                event_id=f"{prepared_job.run_id}-incident-1",
                run_id=prepared_job.run_id,
                step=10,
                category=RuntimeEventCategory.INCIDENT,
                message="incident one",
                payload_ref="artifacts/incident_1.json",
            )
        )
        base.events.append(
            RuntimeEvent(
                event_id=f"{prepared_job.run_id}-incident-2",
                run_id=prepared_job.run_id,
                step=11,
                category=RuntimeEventCategory.INCIDENT,
                message="incident two",
                payload_ref="artifacts/incident_2.json",
            )
        )
        return base


def test_local_process_backend_still_operates_with_config_loading(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="stage4-local", backend=LocalProcessBackend())
    outputs = {result.phase.value: result.output for result in orch.run("stage4-local")}
    assert outputs["runtime_monitor"]["training_outputs"]["status"] == "completed"


def test_stage_profile_changes_runtime_sensitivity(tmp_path: Path) -> None:
    orch = build_orchestrator(
        state_root=tmp_path,
        run_id="stage4-sensitive",
        backend=OneIncidentBackend(),
        stage_name="continuation_repair",
    )
    outputs = {result.phase.value: result.output for result in orch.run("stage4-sensitive")}
    assert outputs["runtime_monitor"]["outcome"] == "soft_suspicion"


def test_stage_strictness_changes_runtime_thresholds(tmp_path: Path) -> None:
    strict_run = build_orchestrator(
        state_root=tmp_path / "strict",
        run_id="stage4-strict-threshold",
        backend=TwoIncidentBackend(),
        stage_name="continuation_repair",
    )
    lenient_run = build_orchestrator(
        state_root=tmp_path / "lenient",
        run_id="stage4-lenient-threshold",
        backend=TwoIncidentBackend(),
        stage_name="early_pretraining",
    )

    strict_outputs = {result.phase.value: result.output for result in strict_run.run("stage4-strict-threshold")}
    lenient_outputs = {result.phase.value: result.output for result in lenient_run.run("stage4-lenient-threshold")}
    assert strict_outputs["runtime_monitor"]["outcome"] == "waste_stop"
    assert lenient_outputs["runtime_monitor"]["outcome"] == "soft_suspicion"


def test_config_loader_reports_json_compatible_requirement(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: not-json\n")
    with pytest.raises(ConfigError, match="JSON-compatible"):
        load_config_file(bad)
