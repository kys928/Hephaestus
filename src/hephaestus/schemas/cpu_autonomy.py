"""Fixed public CPU research comparison; separate from confirmatory #152."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class CpuAutonomyPlan(JsonSchema):
    plan_id: str = "public-tiny-cpu-autonomy.v1"
    seeds: tuple = (1117, 2227, 3337)
    policies: tuple = ("fixed", "heuristic", "adaptive")
    cycles: int = 3
    steps_per_cycle: int = 4
    base_seed: int = 152
    data_seed: int = 20261028
    maximum_program_seconds: int = 240
    initial_learning_rate: float = .02
    fixed_learning_rate: float = .01
    heuristic_schedule: tuple = (.02, .01, .005)
    learning_rate_grid: tuple = (.005, .01, .02, .03, .04)
    development_nll_margin: float = 1e-6
    target_nll_improvement: float = .1

    def __post_init__(self):
        for field in ("seeds", "policies", "heuristic_schedule", "learning_rate_grid"):
            setattr(self, field, tuple(getattr(self, field)))
        if (self.plan_id != "public-tiny-cpu-autonomy.v1" or self.seeds != (1117,2227,3337)
                or self.policies != ("fixed","heuristic","adaptive") or self.cycles != 3 or self.steps_per_cycle != 4
                or self.base_seed != 152 or self.data_seed != 20261028 or self.maximum_program_seconds != 240
                or self.initial_learning_rate != .02 or self.fixed_learning_rate != .01
                or self.heuristic_schedule != (.02,.01,.005) or self.learning_rate_grid != (.005,.01,.02,.03,.04)
                or self.development_nll_margin != 1e-6 or self.target_nll_improvement != .1):
            raise ValueError("changed CPU experiment requires a new prospective plan version")
