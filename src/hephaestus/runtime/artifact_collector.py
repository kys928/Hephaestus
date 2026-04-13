from __future__ import annotations

from pathlib import Path


def collect_artifact_refs(runtime_artifacts: list[str]) -> list[str]:
    return [artifact for artifact in runtime_artifacts if artifact]


def collect_existing_artifacts(artifact_root: str, refs: list[str]) -> tuple[list[str], list[str]]:
    existing: list[str] = []
    missing: list[str] = []
    for ref in refs:
        direct = Path(ref)
        if direct.exists():
            existing.append(ref)
            continue
        fallback = Path(artifact_root) / Path(ref).name
        if fallback.exists():
            existing.append(ref)
        else:
            missing.append(ref)
    return existing, missing
