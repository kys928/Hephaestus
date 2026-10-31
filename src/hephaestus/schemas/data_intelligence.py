from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class DiscoveryBudget(JsonSchema):
    maximum_candidates: int = 8
    maximum_bytes: int = 16 * 1024 * 1024
    maximum_seconds: int = 120

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in (self.maximum_candidates,
                self.maximum_bytes, self.maximum_seconds)):
            raise ValueError("positive finite discovery budgets required")
