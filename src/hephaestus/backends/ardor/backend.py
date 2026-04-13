from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hephaestus.backends.base import BackendRunResult, BackendTarget, PreparedBackendJob
from hephaestus.backends.ardor.dataset_adapter import ArdorDatasetAdapter
from hephaestus.backends.ardor.eval_adapter import ArdorEvalAdapter
from hephaestus.backends.ardor.launcher import ArdorLauncher
from hephaestus.backends.ardor.loader import ArdorLoader
from hephaestus.backends.ardor.runtime_adapter import ArdorRuntimeAdapter
from hephaestus.config_loader import ConfigError, load_named_config


@dataclass(slots=True)
class ArdorBackend:
    name: str = "ardor"
    config_dir: Path = Path("configs")
    loader: ArdorLoader = field(init=False)
    dataset_adapter: ArdorDatasetAdapter = field(init=False)
    eval_adapter: ArdorEvalAdapter = field(init=False)
    launcher: ArdorLauncher = field(init=False)
    runtime_adapter: ArdorRuntimeAdapter = field(init=False)

    def __post_init__(self) -> None:
        self.loader = ArdorLoader(config_dir=self.config_dir)
        self.dataset_adapter = ArdorDatasetAdapter()
        self.eval_adapter = ArdorEvalAdapter()
        self.launcher = ArdorLauncher()
        self.runtime_adapter = ArdorRuntimeAdapter()

    def resolve_target(self, launch_config: dict[str, object]) -> BackendTarget:
        backend_cfg = load_named_config(self.config_dir, "backends", self.name)
        target_cfg = {
            "endpoint": backend_cfg.get("endpoint"),
            "queue": backend_cfg.get("queue"),
            "simulate_only": bool(backend_cfg.get("simulate_only", True)),
            "model_id": str(launch_config.get("parameters", {}).get("model_id", backend_cfg.get("default_model_id", ""))),
        }
        return BackendTarget(backend_name=self.name, dry_run=bool(launch_config.get("dry_run", False)), config=target_cfg)

    def acquire_dataset(self, run_id: str) -> dict[str, object]:
        artifact_ref = Path("artifacts") / run_id / "dataset_manifest.json"
        return {
            "dataset_id": f"ardor-dataset-{run_id}",
            "source_identity": "approved:ardor:manifest",
            "license": "internal",
            "quality_score": 0.9,
            "risks": ["backend_managed"],
            "total_examples": 0,
            "source_ids": ["ardor://managed_dataset"],
            "artifact_ref": str(artifact_ref),
        }

    def preprocess(self, run_id: str) -> dict[str, object]:
        processed_path = Path("artifacts") / run_id / "processed_dataset.jsonl"
        if not processed_path.exists():
            raise ConfigError(f"processed dataset missing for Ardor preprocessing: {processed_path}")
        return {
            "operations": ["validate_manifest", "token_budget_check"],
            "processed_dataset_ref": str(processed_path),
            "dropped_examples": 0,
        }

    def prepare_training_job(
        self,
        *,
        experiment_plan: dict[str, object],
        data_contract: dict[str, object],
        training_plan: dict[str, object],
        launch_config: dict[str, object],
    ) -> PreparedBackendJob:
        target = self.resolve_target(launch_config)
        model = self.loader.resolve_model(str(target.config["model_id"]))
        dataset_input = self.dataset_adapter.adapt(data_contract)
        backend_cfg = load_named_config(self.config_dir, "backends", self.name)

        job = self.launcher.build_prepared_job(
            run_id=str(training_plan["run_id"]),
            artifact_root=str(launch_config["artifact_root"]),
            launch_config=launch_config,
            training_plan=training_plan,
            dataset_input=dataset_input,
            backend_config=backend_cfg,
        )
        job.execution_spec["simulate_only"] = bool(target.config.get("simulate_only", True))
        job.execution_spec["job_spec"]["model"] = {
            "model_id": model.model_id,
            "architecture": model.architecture,
            "tokenizer": model.tokenizer,
        }
        return job

    def launch_training(self, prepared_job: PreparedBackendJob) -> BackendRunResult:
        result = self.runtime_adapter.launch(prepared_job)
        result.intermediate_eval = self.eval_adapter.adapt_intermediate_eval(result.intermediate_eval)
        return result

    def stop(self, run_id: str) -> None:
        return None
