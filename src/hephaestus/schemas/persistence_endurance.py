"""Bounded engineering dose; never a model superiority experiment."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class PersistenceEndurancePlan(JsonSchema):
    schema: str = 'cpu-persistence-endurance.v1'
    cohorts: int = 2
    programs_per_cohort: int = 9
    cycles_per_program: int = 3
    updates_per_cycle: int = 4
    snapshot_restore_after_each_program: bool = True
    fresh_process_cold_model_and_optimizer_replay: bool = True
    nonconfirmatory: bool = True
    paid_resources_authorized: bool = False

    def __post_init__(self):
        if self.to_dict() != dict(schema='cpu-persistence-endurance.v1', cohorts=2,
                programs_per_cohort=9, cycles_per_program=3, updates_per_cycle=4,
                snapshot_restore_after_each_program=True,
                fresh_process_cold_model_and_optimizer_replay=True,
                nonconfirmatory=True, paid_resources_authorized=False):
            raise ValueError('endurance dose and engineering-only scope are frozen')
