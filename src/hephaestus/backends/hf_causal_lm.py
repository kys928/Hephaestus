"""Hugging Face causal-LM backend adapter.

This backend intentionally reuses the local subprocess execution contract: real HF
training is supplied by an explicit trainer script, while this adapter validates
HF-specific configuration and emits backend identity/configuration evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hephaestus.backends.base import BackendRunResult, BackendTarget, PreparedBackendJob
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.config_loader import ConfigError, load_named_config
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


@dataclass(slots=True)
class HFCausalLMBackend(LocalProcessBackend):
    name: str = "hf_causal_lm"
    config_dir: Path = Path("configs")

    def resolve_target(self, launch_config: dict[str, object]) -> BackendTarget:
        cfg = self._backend_config()
        parameters = dict(launch_config.get("parameters", {}))
        target = {
            "profile": str(cfg.get("profile", "hf_local")),
            "device": str(parameters.get("device", cfg.get("device", "cpu"))),
            "model_id": str(parameters.get("model_id", cfg.get("default_model_id", "demo-hf-base"))),
            "trainer_script": str(parameters.get("trainer_script", cfg.get("trainer_script", "tests/fixtures/fake_trainer.py"))),
        }
        return BackendTarget(self.name, bool(launch_config.get("dry_run", False)), target)

    def prepare_training_job(
        self,
        *,
        experiment_plan: dict[str, object],
        data_contract: dict[str, object],
        training_plan: dict[str, object],
        launch_config: dict[str, object],
    ) -> PreparedBackendJob:
        target = self.resolve_target(launch_config)
        self._validate_hf_target(target.config)
        launch_config = dict(launch_config)
        parameters = dict(launch_config.get("parameters", {}))
        parameters.setdefault("trainer_script", str(target.config["trainer_script"]))
        parameters.setdefault("device", str(target.config["device"]))
        parameters.setdefault("model_id", str(target.config["model_id"]))
        launch_config["parameters"] = parameters
        job = super().prepare_training_job(
            experiment_plan=experiment_plan,
            data_contract=data_contract,
            training_plan=training_plan,
            launch_config=launch_config,
        )
        job.backend_name = self.name
        job.execution_spec["hf_config"] = dict(target.config)
        Path(job.artifact_root).mkdir(parents=True, exist_ok=True)
        config_ref = Path(job.artifact_root) / "hf_backend_config.json"
        config_ref.write_text(json.dumps(target.config, indent=2, sort_keys=True))
        job.expected_artifacts.append(str(config_ref))
        return job

    def launch_training(self, prepared_job: PreparedBackendJob) -> BackendRunResult:
        result = super().launch_training(prepared_job)
        config_ref = str(Path(prepared_job.artifact_root) / "hf_backend_config.json")
        if config_ref not in result.artifact_refs and Path(config_ref).exists():
            result.artifact_refs.append(config_ref)
        result.events.append(
            RuntimeEvent(
                event_id=f"{prepared_job.run_id}-hf-config-normalized",
                run_id=prepared_job.run_id,
                step=0,
                category=RuntimeEventCategory.STATUS,
                message="hf_causal_lm_backend_config_normalized",
                payload_ref=config_ref,
            )
        )
        return result

    def _backend_config(self) -> dict[str, object]:
        try:
            return load_named_config(self.config_dir, "backends", self.name)
        except ConfigError:
            return {"profile": "hf_local", "device": "cpu"}

    def _validate_hf_target(self, config: dict[str, object]) -> None:
        device = str(config.get("device", "")).strip()
        if device not in {"cpu", "cuda", "mps"}:
            raise ConfigError(f"hf_causal_lm unsupported device: {device}")
        if not str(config.get("model_id", "")).strip():
            raise ConfigError("hf_causal_lm requires model_id")
        trainer_script = Path(str(config.get("trainer_script", "")))
        if not trainer_script.exists():
            raise ConfigError(f"hf_causal_lm trainer_script missing: {trainer_script}")
