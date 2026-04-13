from __future__ import annotations

import json
from pathlib import Path


class MetricsArtifactError(ValueError):
    pass


def read_metrics(intermediate_artifact: dict[str, object]) -> dict[str, float]:
    metrics_ref = str(intermediate_artifact.get("metrics_ref", ""))
    has_inline = "probe_score" in intermediate_artifact and "toxicity" in intermediate_artifact

    if metrics_ref:
        metrics_path = Path(metrics_ref)
        if metrics_path.exists() and metrics_path.is_file():
            payload = json.loads(metrics_path.read_text())
            return {
                "probe_score": float(payload.get("probe_score", 0.0)),
                "toxicity": float(payload.get("toxicity", 1.0)),
            }
        if not has_inline:
            raise MetricsArtifactError(f"metrics artifact missing or unreadable: {metrics_ref}")

    if has_inline:
        return {
            "probe_score": float(intermediate_artifact.get("probe_score", 0.0)),
            "toxicity": float(intermediate_artifact.get("toxicity", 1.0)),
        }

    raise MetricsArtifactError("no metrics artifact reference or inline metrics provided")
