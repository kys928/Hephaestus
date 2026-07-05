"""Deterministic UTC time helpers for persisted records."""

from __future__ import annotations

from datetime import datetime, timezone

UTC_SUFFIX = "Z"


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string ending in ``Z``."""

    return format_utc(datetime.now(timezone.utc))


def format_utc(value: datetime) -> str:
    """Normalize a datetime to second-precision UTC JSON text."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc).replace(microsecond=0)
    return value.isoformat().replace("+00:00", UTC_SUFFIX)


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp, accepting a trailing ``Z``."""

    normalized = value.strip()
    if normalized.endswith(UTC_SUFFIX):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
