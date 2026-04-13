from __future__ import annotations


def stop_recommendation(monitor_outcome: str) -> str:
    if monitor_outcome == "hard_abort":
        return "stop_now"
    if monitor_outcome == "waste_stop":
        return "stop_for_waste"
    if monitor_outcome == "soft_suspicion":
        return "pause_and_review"
    return "continue"
