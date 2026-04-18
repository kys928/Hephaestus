from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.config_loader import ConfigError, load_named_config
from hephaestus.schemas.stage_profile import StageProfile


@dataclass(slots=True)
class StagePolicy:
    config_dir: Path = Path("configs")

    def resolve(self, stage_name: str) -> StageProfile:
        payload = load_named_config(self.config_dir, "stage_profiles", stage_name)
        return self._from_payload(stage_name, payload)

    def _from_payload(self, stage_name: str, payload: dict[str, object]) -> StageProfile:
        required = ("strictness", "eval_pack", "deterministic_gates")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ConfigError(f"stage profile '{stage_name}' missing fields: {', '.join(missing)}")

        gates = payload["deterministic_gates"]
        if not isinstance(gates, dict):
            raise ConfigError(f"stage profile '{stage_name}' deterministic_gates must be an object")
        if "min_probe_score" not in gates or "max_toxicity" not in gates:
            raise ConfigError(f"stage profile '{stage_name}' requires min_probe_score and max_toxicity gates")

        allowed = payload.get("allowed_next_actions", [])
        if not isinstance(allowed, list):
            raise ConfigError(f"stage profile '{stage_name}' allowed_next_actions must be a list")

        cert_profile = payload.get("certification_profile", {})
        if not isinstance(cert_profile, dict):
            raise ConfigError(f"stage profile '{stage_name}' certification_profile must be an object")
        min_consistent_runs = int(cert_profile.get("min_consistent_runs", 1))
        if min_consistent_runs < 1:
            raise ConfigError(f"stage profile '{stage_name}' certification_profile.min_consistent_runs must be >= 1")

        return StageProfile(
            name=str(payload.get("name", stage_name)),
            strictness=str(payload["strictness"]),
            eval_pack=str(payload["eval_pack"]),
            deterministic_gates={
                "min_probe_score": float(gates["min_probe_score"]),
                "max_toxicity": float(gates["max_toxicity"]),
            },
            allowed_next_actions=[str(item) for item in allowed],
            certification_profile={
                "eligibility": str(cert_profile.get("eligibility", "standard")),
                "require_recheck": bool(cert_profile.get("require_recheck", False)),
                "min_consistent_runs": min_consistent_runs,
            },
        )
