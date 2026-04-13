from __future__ import annotations


def read_metrics(intermediate_artifact: dict[str, object]) -> dict[str, float]:
    return {
        "probe_score": float(intermediate_artifact.get("probe_score", 0.0)),
        "toxicity": float(intermediate_artifact.get("toxicity", 1.0)),
    }
