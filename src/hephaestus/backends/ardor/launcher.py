from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from hephaestus.backends.base import PreparedBackendJob
from hephaestus.config_loader import ConfigError
from hephaestus.runtime.launcher import launch_subprocess


_SUPPORTED_EXECUTION_MODES = {"local_process"}


@dataclass(slots=True)
class ArdorLaunchOutcome:
    run_id: str
    status: str
    execution_mode: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    contract_ref: str | None = None


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

        manifest = dataset_input.get("dataset_manifest")
        if not isinstance(manifest, dict):
            raise ConfigError("Ardor launcher requires dataset_input.dataset_manifest")
        processed_dataset_ref = str(manifest.get("path", "")).strip()
        if not processed_dataset_ref:
            raise ConfigError("Ardor launcher requires dataset_manifest.path")

        max_steps = int(training_plan.get("max_steps", 0))
        if max_steps <= 0:
            raise ConfigError("Ardor launcher requires positive max_steps")

        parameters = dict(launch_config.get("parameters", {}))
        execution_mode = str(parameters.get("ardor_execution_mode") or backend_config.get("execution_mode") or "").strip()
        if not execution_mode:
            raise ConfigError("Ardor backend config requires execution_mode")

        if execution_mode not in _SUPPORTED_EXECUTION_MODES:
            raise ConfigError(f"Ardor execution_mode '{execution_mode}' is unsupported")

        local_runner_path = str(parameters.get("ardor_runner_script") or backend_config.get("local_runner_path") or "").strip()
        if not local_runner_path:
            raise ConfigError("Ardor local_process mode requires explicit local_runner_path or parameters.ardor_runner_script")

        execution_spec = {
            "runner": "ardor_local_process",
            "submission_kind": "local_subprocess",
            "execution_mode": execution_mode,
            "endpoint": endpoint,
            "queue": queue,
            "job_spec": {
                "run_id": run_id,
                "artifact_root": artifact_root,
                "dataset": dataset_input,
                "max_steps": max_steps,
                "eval_every_steps": int(training_plan.get("eval_every_steps", 0)),
                "checkpoint_every_steps": int(training_plan.get("checkpoint_every_steps", 0)),
                "runner_script": local_runner_path,
                "parameters": parameters,
            },
        }
        return PreparedBackendJob(
            run_id=run_id,
            backend_name="ardor",
            artifact_root=artifact_root,
            expected_artifacts=[f"{artifact_root}/ardor_runtime_contract.json"],
            execution_spec=execution_spec,
        )

    def launch(self, prepared_job: PreparedBackendJob) -> ArdorLaunchOutcome:
        execution_mode = str(prepared_job.execution_spec.get("execution_mode", "")).strip()
        if execution_mode not in _SUPPORTED_EXECUTION_MODES:
            return ArdorLaunchOutcome(
                run_id=prepared_job.run_id,
                status="unsupported_execution_mode",
                execution_mode=execution_mode,
                stderr=f"unsupported execution_mode={execution_mode}",
            )

        job_spec = dict(prepared_job.execution_spec.get("job_spec", {}))
        parameters = dict(job_spec.get("parameters", {}))
        runner_script = str(job_spec.get("runner_script", "")).strip()
        if not runner_script:
            return ArdorLaunchOutcome(
                run_id=prepared_job.run_id,
                status="launch_failed",
                execution_mode=execution_mode,
                stderr="missing runner_script in execution spec",
            )

        artifact_root = Path(prepared_job.artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            runner_script,
            "--run-id",
            prepared_job.run_id,
            "--artifact-root",
            str(artifact_root),
            "--dataset-ref",
            str(dict(dict(job_spec.get("dataset", {})).get("dataset_manifest", {})).get("path", "")),
            "--max-steps",
            str(job_spec.get("max_steps", 0)),
            "--contract-path",
            str(artifact_root / "ardor_runtime_contract.json"),
        ]
        for flag_name, cli_flag in (
            ("ardor_fail_launch", "--fail-launch"),
            ("ardor_fail_runtime", "--fail-runtime"),
            ("ardor_omit_metrics", "--omit-metrics"),
            ("ardor_omit_checkpoint", "--omit-checkpoint"),
            ("ardor_malformed_contract", "--malformed-contract"),
            ("ardor_unsupported_state", "--unsupported-state"),
        ):
            if str(parameters.get(flag_name, "0")) == "1":
                command.append(cli_flag)

        try:
            launch = launch_subprocess(
                command,
                cwd=os.getcwd(),
                env={**os.environ, "HEPHAESTUS_RUN_ID": prepared_job.run_id},
            )
        except OSError as exc:
            return ArdorLaunchOutcome(
                run_id=prepared_job.run_id,
                status="launch_failed",
                execution_mode=execution_mode,
                stderr=str(exc),
            )

        contract_ref = str(artifact_root / "ardor_runtime_contract.json")
        if launch.returncode != 0 and not Path(contract_ref).exists():
            return ArdorLaunchOutcome(
                run_id=prepared_job.run_id,
                status="launch_failed",
                execution_mode=execution_mode,
                returncode=launch.returncode,
                stdout=launch.stdout,
                stderr=launch.stderr,
                contract_ref=contract_ref,
            )

        return ArdorLaunchOutcome(
            run_id=prepared_job.run_id,
            status="launched",
            execution_mode=execution_mode,
            returncode=launch.returncode,
            stdout=launch.stdout,
            stderr=launch.stderr,
            contract_ref=contract_ref,
        )
