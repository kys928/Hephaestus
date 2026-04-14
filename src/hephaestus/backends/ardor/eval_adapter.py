from __future__ import annotations

from dataclasses import dataclass

from hephaestus.config_loader import ConfigError


@dataclass(slots=True)
class ArdorEvalAdapter:
    def adapt_intermediate_eval(self, payload: dict[str, object]) -> dict[str, object]:
        supported = {"metrics_ref", "probe_ref", "deterministic_ref", "runtime_log_ref", "probe_score", "toxicity"}
        unsupported = sorted(key for key in payload if key not in supported)
        if unsupported:
            raise ConfigError(f"Ardor eval adapter unsupported fields: {', '.join(unsupported)}")
        return {key: value for key, value in payload.items() if key in supported}
