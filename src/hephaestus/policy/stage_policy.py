from __future__ import annotations

from dataclasses import dataclass

from hephaestus.schemas.stage_profile import StageProfile


_DEFAULTS = {
    "early_pretraining": StageProfile(
        name="early_pretraining",
        strictness="lenient",
        eval_pack="pretraining_probes",
        deterministic_gates={"max_toxicity": 0.2, "min_probe_score": 0.55},
        allowed_next_actions=["continue_lineage_best", "rerun_same_config", "reject_checkpoint"],
    ),
    "repair": StageProfile(
        name="repair",
        strictness="strict",
        eval_pack="continuation_repair",
        deterministic_gates={"max_toxicity": 0.05, "min_probe_score": 0.75},
        allowed_next_actions=["promote_checkpoint", "rollback_to_checkpoint", "reject_checkpoint", "abort_run"],
    ),
}


@dataclass(slots=True)
class StagePolicy:
    def resolve(self, stage_name: str) -> StageProfile:
        return _DEFAULTS.get(stage_name, _DEFAULTS["early_pretraining"])
