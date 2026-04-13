from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from hephaestus.backends.base import BackendRunResult, BackendTarget, PreparedBackendJob
from hephaestus.runtime.artifact_collector import collect_existing_artifacts
from hephaestus.runtime.event_stream import events_from_process_output
from hephaestus.runtime.health_checks import process_failed
from hephaestus.runtime.launcher import launch_subprocess


@dataclass(slots=True)
class LocalProcessBackend:
    name: str = "local_process"

    def resolve_target(self, launch_config: dict[str, object]) -> BackendTarget:
        return BackendTarget(backend_name=self.name, dry_run=bool(launch_config.get("dry_run", False)), config=dict(launch_config))

    def acquire_dataset(self, run_id: str) -> dict[str, object]:
        artifact_root = Path("artifacts") / run_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        dataset_path = artifact_root / "dataset_manifest.json"
        dataset_payload = {
            "dataset_id": f"dataset-{run_id}",
            "source_identity": "approved:local:test_fixture",
            "license": "internal",
            "quality_score": 0.9,
            "risks": ["small_fixture"],
            "total_examples": 4,
            "source_ids": ["local://fixtures/tiny_corpus"],
            "artifact_ref": str(dataset_path),
        }
        dataset_path.write_text(json.dumps(dataset_payload, indent=2))
        return dataset_payload

    def preprocess(self, run_id: str) -> dict[str, object]:
        artifact_root = Path("artifacts") / run_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        processed_path = artifact_root / "processed_dataset.jsonl"
        processed_path.write_text('{"text":"example"}\n')
        return {
            "operations": ["normalize", "filter", "chunk"],
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
        run_id = str(training_plan["run_id"])
        artifact_root = Path(str(launch_config["artifact_root"]))
        artifact_root.mkdir(parents=True, exist_ok=True)
        script = str(launch_config["parameters"].get("trainer_script", "tests/fixtures/fake_trainer.py"))
        command = [
            sys.executable,
            script,
            "--run-id",
            run_id,
            "--artifact-root",
            str(artifact_root),
            "--dataset-ref",
            str(data_contract["processed_dataset_ref"]),
            "--max-steps",
            str(training_plan["max_steps"]),
        ]
        if launch_config["parameters"].get("force_fail") == "1":
            command.append("--fail")
        if launch_config["parameters"].get("omit_metrics") == "1":
            command.append("--omit-metrics")
        return PreparedBackendJob(
            run_id=run_id,
            backend_name=self.name,
            artifact_root=str(artifact_root),
            expected_artifacts=[
                str(artifact_root / "metrics.json"),
                str(artifact_root / "probe.json"),
                str(artifact_root / "deterministic.json"),
                str(artifact_root / "checkpoint_step_200.ckpt"),
            ],
            execution_spec={
                "runner": "subprocess",
                "command": command,
                "cwd": os.getcwd(),
                "env": {**os.environ, "HEPHAESTUS_RUN_ID": run_id},
            },
        )

    def launch_training(self, prepared_job: PreparedBackendJob) -> BackendRunResult:
        spec = dict(prepared_job.execution_spec)
        launch = launch_subprocess(
            list(spec.get("command", [])),
            cwd=str(spec.get("cwd")) if spec.get("cwd") else None,
            env=dict(spec.get("env", {})) or None,
        )
        events = events_from_process_output(prepared_job.run_id, launch.stdout, launch.stderr)
        existing, missing = collect_existing_artifacts(prepared_job.artifact_root, prepared_job.expected_artifacts)
        status = "completed"
        if process_failed(launch.returncode) or missing:
            status = "failed"
        checkpoint_candidates = [{"checkpoint_ref": ref, "probe_score": 0.7} for ref in existing if ref.endswith(".ckpt")]
        intermediate_eval = {
            "metrics_ref": next((ref for ref in existing if ref.endswith("metrics.json")), ""),
            "probe_ref": next((ref for ref in existing if ref.endswith("probe.json")), ""),
            "deterministic_ref": next((ref for ref in existing if ref.endswith("deterministic.json")), ""),
        }
        return BackendRunResult(
            run_id=prepared_job.run_id,
            status=status,
            events=events,
            artifact_refs=existing,
            checkpoint_candidates=checkpoint_candidates,
            intermediate_eval=intermediate_eval,
        )

    def stop(self, run_id: str) -> None:
        return None
