from __future__ import annotations


def collect_artifact_refs(runtime_artifacts: list[str]) -> list[str]:
    return [artifact for artifact in runtime_artifacts if artifact]
