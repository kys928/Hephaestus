from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.config_loader import ConfigError, load_named_config


@dataclass(slots=True)
class ArdorModelResolution:
    model_id: str
    architecture: str
    tokenizer: str


@dataclass(slots=True)
class ArdorLoader:
    config_dir: Path = Path("configs")

    def resolve_model(self, model_id: str) -> ArdorModelResolution:
        registry = load_named_config(self.config_dir, "", "model_registry")
        models = registry.get("models")
        if not isinstance(models, dict) or model_id not in models:
            raise ConfigError(f"model '{model_id}' is missing in model registry")
        model = models[model_id]
        if not isinstance(model, dict):
            raise ConfigError(f"model registry entry '{model_id}' must be an object")

        required = ("architecture", "tokenizer")
        missing = [key for key in required if key not in model]
        if missing:
            raise ConfigError(f"model '{model_id}' missing metadata: {', '.join(missing)}")

        architecture = str(model["architecture"])
        if architecture != "hf_causal_lm":
            raise ConfigError(f"Ardor only supports architecture=hf_causal_lm, got '{architecture}'")
        return ArdorModelResolution(model_id=model_id, architecture=architecture, tokenizer=str(model["tokenizer"]))

    def validate_checkpoint_metadata(self, checkpoint: dict[str, object], expected_architecture: str) -> str:
        checkpoint_ref = str(checkpoint.get("checkpoint_ref", ""))
        if not checkpoint_ref:
            raise ConfigError("checkpoint metadata missing checkpoint_ref")

        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise ConfigError(f"checkpoint '{checkpoint_ref}' missing metadata object")
        if str(metadata.get("architecture", "")) != expected_architecture:
            raise ConfigError(f"checkpoint '{checkpoint_ref}' architecture mismatch")
        return checkpoint_ref
