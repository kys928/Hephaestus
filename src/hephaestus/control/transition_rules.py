from __future__ import annotations

from hephaestus.control.spine import SPINE_ORDER, SpinePhase


def next_phase(current: SpinePhase) -> SpinePhase | None:
    try:
        index = SPINE_ORDER.index(current)
    except ValueError:
        return None
    if index + 1 >= len(SPINE_ORDER):
        return None
    return SPINE_ORDER[index + 1]
