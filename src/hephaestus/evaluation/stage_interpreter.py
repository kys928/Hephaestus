from __future__ import annotations


def interpret_stage(strictness: str, deterministic_passed: bool) -> tuple[float, str]:
    if deterministic_passed:
        return (0.8 if strictness == "strict" else 0.65, "none")
    if strictness == "strict":
        return (0.3, "deterministic_regression")
    return (0.45, "instability")
