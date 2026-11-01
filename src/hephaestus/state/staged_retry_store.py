from dataclasses import dataclass
from hephaestus.schemas.staged_retry import StagedRetry
from hephaestus.storage.base import StateRepository

COLLECTION = "production_staged_attempts"


@dataclass(slots=True)
class StagedRetryStore:
    repository: StateRepository

    def current(self, cycle_key):
        row = self.repository.get_latest(COLLECTION, "cycle_key", cycle_key)
        return StagedRetry.from_dict(row) if row else None

    def save(self, row: StagedRetry):
        previous = self.current(row.cycle_key)
        if previous:
            if previous.to_dict() == row.to_dict():
                return
            if (row.attempt != previous.attempt + 1 or row.maximum_attempts != previous.maximum_attempts
                    or row.predecessor_run_id != previous.run_id):
                raise ValueError("staged attempt ancestry or retry ceiling changed")
        elif row.attempt != 1:
            raise ValueError("first staged attempt must be one")
        self.repository.append(COLLECTION, row.to_dict())
