"""Schema: JudgeExit with finite next-action space."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._base import JsonSchema


class JudgeExitAction(str, Enum):
    """Finite judge-exit actions allowed by the control model."""

    CONTINUE_LINEAGE_BEST = "continue_lineage_best"
    CONTINUE_FROM_CHECKPOINT = "continue_from_checkpoint"
    RERUN_SAME_CONFIG = "rerun_same_config"
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"
    BRANCH_NEW_EXPERIMENT = "branch_new_experiment"
    RESTART_LINEAGE = "restart_lineage"
    ABORT_RUN = "abort_run"
    PROMOTE_CHECKPOINT = "promote_checkpoint"
    REJECT_CHECKPOINT = "reject_checkpoint"


@dataclass(slots=True)
class JudgeExit(JsonSchema):
    run_id: str
    verdict: str
    next_action: JudgeExitAction
