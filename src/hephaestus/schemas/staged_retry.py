"""A new attempt preserves its failed predecessor and scientific identity."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class StagedRetry(JsonSchema):
    cycle_key: str
    attempt: int
    run_id: str
    maximum_attempts: int
    predecessor_run_id: str | None = None
    failure_ref: str | None = None
    decision_ref: str | None = None
    scientific_fingerprint: str | None = None
    not_before: str | None = None

    def __post_init__(self):
        if (type(self.attempt) is not int or type(self.maximum_attempts) is not int
                or not 1 <= self.attempt <= self.maximum_attempts <= 5):
            raise ValueError("staged retry attempts must be bounded to 1..5")
