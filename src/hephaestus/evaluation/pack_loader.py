from __future__ import annotations

from pathlib import Path

from hephaestus.config_loader import ConfigError, load_named_config


def load_eval_pack(pack_name: str, config_dir: Path = Path("configs")) -> dict[str, object]:
    payload = load_named_config(config_dir, "eval_packs", pack_name)
    required = ("pack_name", "required_metrics")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ConfigError(f"eval pack '{pack_name}' missing fields: {', '.join(missing)}")
    metrics = payload["required_metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError(f"eval pack '{pack_name}' required_metrics must be a non-empty list")
    return {
        "pack_name": str(payload["pack_name"]),
        "required_metrics": [str(metric) for metric in metrics],
        "description": str(payload.get("description", "")),
        "supports_generation_signals": bool(payload.get("supports_generation_signals", False)),
    }
