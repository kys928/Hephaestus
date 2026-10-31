"""Versioned finite-epoch optimizer windows, independent of model libraries."""
import math
from fractions import Fraction

SCHEDULE_VERSION = "epoch-bounded-microbatch-mean.v2"


def batch_limit(batch_count: int, max_epochs: float | None) -> int | None:
    if type(batch_count) is not int or batch_count < 1:
        raise ValueError("positive batch count required")
    if max_epochs is None:
        return None
    if isinstance(max_epochs, bool) or not math.isfinite(max_epochs) or max_epochs <= 0:
        raise ValueError("positive finite epoch limit required")
    # Fractional epochs mean a ceiling on the fraction of batches, so at most
    # one indivisible final batch is rounded up. No modulo wrap past this cap.
    return math.ceil(batch_count * Fraction(str(max_epochs)))


def step_limit(batch_count: int, accumulation: int, maximum_steps: int,
               max_epochs: float | None) -> int:
    if any(type(n) is not int or n < 1 for n in (accumulation, maximum_steps)):
        raise ValueError("positive integer accumulation and step ceiling required")
    limit = batch_limit(batch_count, max_epochs)
    if limit is None:
        return maximum_steps
    full, remainder = divmod(limit, batch_count)
    windows = full * math.ceil(batch_count / accumulation) + math.ceil(remainder / accumulation)
    return min(maximum_steps, windows)


def cursor_at_step(batch_count: int, accumulation: int, step: int,
                   max_epochs: float | None) -> int:
    if (type(accumulation) is not int or accumulation < 1
            or type(step) is not int or step < 0):
        raise ValueError("nonnegative integer step and positive accumulation required")
    batch_limit(batch_count, max_epochs)
    full, partial = divmod(step, math.ceil(batch_count / accumulation))
    cursor = full * batch_count + min(partial * accumulation, batch_count)
    limit = batch_limit(batch_count, max_epochs)
    return min(cursor, limit) if limit is not None else cursor


def training_windows(batch_count: int, accumulation: int, maximum_steps: int,
                     max_epochs: float | None):
    limit = batch_limit(batch_count, max_epochs)
    consumed = 0
    steps = 0
    while steps < step_limit(batch_count, accumulation, maximum_steps, max_epochs):
        width = batch_count if limit is None else min(batch_count, limit - consumed)
        for offset in range(0, width, accumulation):
            if steps == maximum_steps:
                return
            indexes = tuple(range(offset, min(offset + accumulation, width)))
            yield indexes
            consumed += len(indexes)
            steps += 1
