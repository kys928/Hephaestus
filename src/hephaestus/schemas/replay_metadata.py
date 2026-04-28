from __future__ import annotations


def build_replay_metadata(
    checkpoint_ref: str | None,
    checkpoint_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = checkpoint_evidence or {}
    integrity_level = str(evidence.get("integrity_level", "ref") or "ref")
    content_hash = str(evidence.get("content_hash", "") or "")

    has_hash = bool(content_hash)
    hash_verified = integrity_level == "content_hash" and has_hash

    if hash_verified:
        return {
            "checkpoint_integrity_level": "content_hash",
            "requires_checkpoint_ref_match": bool(checkpoint_ref),
            "requires_content_hash_match": True,
            "content_hash_available": True,
            "checkpoint_content_hash": content_hash,
            "replay_scope": "content_hash_verified",
            "replay_claim": (
                "Replay requires matching recorded checkpoint reference, recorded checkpoint content hash, "
                "and recorded run/config/eval metadata."
            ),
            "limitations": [],
        }

    limitations = []
    if checkpoint_ref:
        limitations.append("reference-level evidence does not prove byte-identical checkpoint contents")
    if not has_hash:
        limitations.append("checkpoint content hash was not recorded")

    return {
        "checkpoint_integrity_level": "ref",
        "requires_checkpoint_ref_match": bool(checkpoint_ref),
        "requires_content_hash_match": False,
        "content_hash_available": False,
        "replay_scope": "reference_only",
        "replay_claim": (
            "Replay requires matching recorded checkpoint reference and recorded run/config/eval metadata. "
            "Full content-hash replay is not guaranteed unless content hashes are recorded."
        ),
        "limitations": limitations,
    }
