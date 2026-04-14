from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ._base import JsonSchema


class JudgeEntryMode(str, Enum):
    CONTINUE_LINEAGE_BEST = "continue_lineage_best"
    CONTINUE_FROM_CHECKPOINT = "continue_from_checkpoint"
    CONTINUE_FROM_LAST_STABLE = "continue_from_last_stable"
    RERUN_SAME_CONFIG = "rerun_same_config"
    BRANCH_FROM_CHECKPOINT = "branch_from_checkpoint"
    RESTART_LINEAGE = "restart_lineage"


@dataclass(slots=True)
class JudgeEntry(JsonSchema):
    run_id: str
    lineage_id: str
    objective: str
    constraints: list[str] = field(default_factory=list)
    approved: bool = True
    stage_name: str = ""
    entry_mode: JudgeEntryMode = JudgeEntryMode.RERUN_SAME_CONFIG
