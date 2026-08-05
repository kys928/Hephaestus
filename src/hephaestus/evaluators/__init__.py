"""Dependency-injected evaluator adapters.

Adapters in this package may add evidence, but they do not own deterministic
gates or promotion decisions.
"""

from .judge_adapter import JudgeModelAdapter

__all__ = ["JudgeModelAdapter"]
