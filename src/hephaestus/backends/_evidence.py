from __future__ import annotations

from pathlib import Path

from hephaestus.utils.hashing import hash_file


def build_artifact_evidence(ref: str) -> dict[str, object]:
    path = Path(ref)
    exists = path.exists()
    evidence: dict[str, object] = {"ref": ref, "exists": exists}
    if exists and path.is_file():
        evidence.update({"byte_size": path.stat().st_size, "hash_type": "sha256", "content_hash": hash_file(path)})
    return evidence


def summarize_artifact_evidence(refs: list[str]) -> dict[str, object]:
    artifacts = [build_artifact_evidence(ref) for ref in refs]
    return {
        "artifacts": artifacts,
        "present_count": sum(1 for item in artifacts if item.get("exists")),
        "missing_count": sum(1 for item in artifacts if not item.get("exists")),
    }
