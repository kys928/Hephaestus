from __future__ import annotations


def select_checkpoint(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        return {"checkpoint_ref": "", "score": 0.0, "reason": "missing_checkpoints"}

    if len(candidates) == 1:
        only = dict(candidates[0])
        only.setdefault("reason", "single_checkpoint")
        return only

    scored = [candidate for candidate in candidates if "probe_score" in candidate]
    if len(scored) != len(candidates):
        return {"checkpoint_ref": "", "score": 0.0, "reason": "inconclusive_multiple_checkpoints"}

    best = max(scored, key=lambda item: float(item.get("probe_score", 0.0)))
    selected = dict(best)
    selected.setdefault("reason", "best_probe_score")
    return selected
