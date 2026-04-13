from __future__ import annotations


def select_checkpoint(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        return {"checkpoint_ref": "", "score": 0.0}
    return max(candidates, key=lambda item: float(item.get("probe_score", 0.0)))
