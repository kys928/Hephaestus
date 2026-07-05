from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_deterministic(fn: Callable[[], T], *, attempts: int = 1) -> T:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry helper preserves final exception
            last_error = exc
    assert last_error is not None
    raise last_error
