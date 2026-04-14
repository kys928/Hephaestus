from __future__ import annotations

import json
from pathlib import Path


class MetricsArtifactError(ValueError):
    pass


def _extract(payload: dict[str, object]) -> dict[str, float] | None:
    if "probe_score" in payload and "toxicity" in payload:
        return {"probe_score": float(payload["probe_score"]), "toxicity": float(payload["toxicity"])}

    nested = payload.get("metrics")
    if isinstance(nested, dict) and "probe_score" in nested and "toxicity" in nested:
        return {"probe_score": float(nested["probe_score"]), "toxicity": float(nested["toxicity"])}

    return None


def read_metrics(intermediate_artifact: dict[str, object]) -> dict[str, float]:
    metrics_ref = str(intermediate_artifact.get("metrics_ref", ""))
    inline = _extract(intermediate_artifact)

    if metrics_ref:
        metrics_path = Path(metrics_ref)
        if metrics_path.exists() and metrics_path.is_file():
            payload = json.loads(metrics_path.read_text())
            if isinstance(payload, dict) and (extracted := _extract(payload)):
                return extracted
            raise MetricsArtifactError(f"metrics artifact missing required fields: {metrics_ref}")
        if inline is None:
            raise MetricsArtifactError(f"metrics artifact missing or unreadable: {metrics_ref}")

    if inline is not None:
        return inline

    raise MetricsArtifactError("no metrics artifact reference or inline metrics provided")
