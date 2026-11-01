"""Producer evidence; absence or an in-flight update never means zero progress."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class OptimizerProgress(JsonSchema):
    run_id: str
    config_fingerprint: str
    optimizer_steps: int
    optimizer_update_started: bool
    phase: str
    schema: str = "hf-optimizer-progress.v1"

    def __post_init__(self):
        if (self.schema!="hf-optimizer-progress.v1" or not self.run_id or not self.config_fingerprint
                or type(self.optimizer_steps) is not int or self.optimizer_steps<0
                or type(self.optimizer_update_started) is not bool
                or self.phase not in {"pre_optimizer","update_in_flight","updated"}
                or (self.phase!="pre_optimizer" and not self.optimizer_update_started)):
            raise ValueError("invalid optimizer progress receipt")
