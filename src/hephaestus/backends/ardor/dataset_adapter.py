from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.config_loader import ConfigError


@dataclass(slots=True)
class ArdorDatasetAdapter:
    def adapt(self, data_contract: dict[str, object]) -> dict[str, object]:
        required = ("processed_dataset_ref", "schema_version", "min_tokens")
        missing = [key for key in required if key not in data_contract]
        if missing:
            raise ConfigError(f"trainable data contract missing fields: {', '.join(missing)}")

        dataset_ref = Path(str(data_contract["processed_dataset_ref"]))
        if not dataset_ref.exists():
            raise ConfigError(f"processed dataset missing for Ardor launch: {dataset_ref}")

        if str(data_contract["schema_version"]) != "v1":
            raise ConfigError("Ardor dataset adapter only supports schema_version=v1")

        return {
            "dataset_manifest": {
                "format": "jsonl",
                "path": str(dataset_ref),
                "min_tokens": int(data_contract["min_tokens"]),
            }
        }
