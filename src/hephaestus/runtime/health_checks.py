from __future__ import annotations

from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


def count_incidents(events: list[RuntimeEvent]) -> int:
    return len([event for event in events if event.category is RuntimeEventCategory.INCIDENT])


def count_deterministic_failures(events: list[RuntimeEvent]) -> int:
    failures = 0
    for event in events:
        if event.category is RuntimeEventCategory.DETERMINISTIC_CHECK and "fail" in event.message:
            failures += 1
    return failures


def process_failed(returncode: int) -> bool:
    return returncode != 0
