from __future__ import annotations


def reward_score(helpfulness: float, harmlessness: float) -> float:
    return round((helpfulness * 0.7) + (harmlessness * 0.3), 4)
