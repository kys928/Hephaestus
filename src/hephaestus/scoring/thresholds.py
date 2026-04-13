from __future__ import annotations


def check_threshold(metric_value: float, minimum: float) -> bool:
    return metric_value >= minimum
