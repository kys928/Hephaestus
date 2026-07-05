"""Deterministic retry helpers for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    delay_seconds: float = 0.0
    retryable_errors: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")


@dataclass(slots=True)
class RetryError(RuntimeError):
    message: str
    attempts: list[dict[str, object]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


def run_with_retry(operation: Callable[[], T], policy: RetryPolicy | None = None) -> T:
    """Run an operation with fixed, deterministic retry behavior."""

    retry_policy = policy or RetryPolicy()
    attempts: list[dict[str, object]] = []
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            return operation()
        except retry_policy.retryable_errors as exc:
            attempts.append({"attempt": attempt, "error_type": type(exc).__name__, "message": str(exc)})
            if attempt == retry_policy.max_attempts:
                raise RetryError("operation failed after deterministic retries", attempts) from exc
            if retry_policy.delay_seconds:
                sleep(retry_policy.delay_seconds)
    raise RetryError("operation failed without executing", attempts)
