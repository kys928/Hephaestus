from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.base import PreparedBackendJob
from hephaestus.config_loader import ConfigError


@dataclass(slots=True)
class ArdorLauncher:
    def build_prepared_job(
        self,
        *,
        run_id: str,
        artifact_root: str,
        launch_config: dict[str, object],
        training_plan: dict[str, object],
        dataset_input: dict[str, object],
        backend_config: dict[str, object],
    ) -> PreparedBackendJob:
        if str(launch_config.get("backend", "")) != "ardor":
            raise ConfigError("Ardor launcher requires backend='ardor'")

        endpoint = str(backend_config.get("endpoint", "")).strip()
        queue = str(backend_config.get("queue", "")).strip()
        if not endpoint or not queue:
            raise ConfigError("Ardor backend config requires endpoint and queue")

        max_steps = int(training_plan.get("max_steps", 0))
        if max_steps <= 0:
            raise ConfigError("Ardor launcher requires positive max_steps")

        execution_spec = {
            "runner": "ardor_api",
            "endpoint": endpoint,
            "queue": queue,
            "job_spec": {
                "run_id": run_id,
                "artifact_root": artifact_root,
                "dataset": dataset_input,
                "max_steps": max_steps,
                "eval_every_steps": int(training_plan.get("eval_every_steps", 0)),
                "checkpoint_every_steps": int(training_plan.get("checkpoint_every_steps", 0)),
                "parameters": dict(launch_config.get("parameters", {})),
            },
        }
        return PreparedBackendJob(
            run_id=run_id,
            backend_name="ardor",
            artifact_root=artifact_root,
            expected_artifacts=[
                f"{artifact_root}/metrics.json",
                f"{artifact_root}/probe.json",
                f"{artifact_root}/deterministic.json",
            ],
            execution_spec=execution_spec,
        )
