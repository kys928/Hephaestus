"""Nonpromotion action application MUST record, never execute, a Judge continuation."""
from types import SimpleNamespace
import pytest
from hephaestus.roles.cpu_research_evaluation import CpuResearchActionRole


class OnlyAppendState:
    def __init__(self):
        self.records=[]
    def put(self, operation_id, input_ref, payload):
        self.records.append((operation_id,payload))
        return "sha256:"+"f"*64


class Artifact:
    def put_bytes(self, data, *, media_type):
        return SimpleNamespace(artifact_ref="sha256:"+"a"*64)


def run(action):
    state=OnlyAppendState()
    context=SimpleNamespace(source=lambda *args:{"verdict":"approved","next_action":action},
                            artifacts=Artifact(),state=state)
    req=SimpleNamespace(substep="action_application",workflow_id="w",run_id="r",operation_id="op",
       prior_outputs={"judge_exit.action_boundary":{"proposed_action":action}})
    return CpuResearchActionRole(context).execute(req),state


def test_measured_improvement_recommendation_is_recorded_but_never_executed():
    result,state=run("continue_from_checkpoint")
    payload=result.records[0].payload
    assert payload["action"]=="continue_from_checkpoint"
    assert payload["human_review_still_required"] is True
    assert payload["model_promoted"] is False
    assert payload["baseline_retained"] is True
    assert payload["checkpoint_activated"] is False
    assert payload["continuation_executed"] is False
    assert len(state.records)==1
    assert state.records[0][1]["effect"]=="recorded_research_disposition_only"


def test_actual_promotion_or_rollback_remains_denied():
    for action in ("promote_checkpoint","rollback_to_checkpoint","branch_new_experiment"):
        with pytest.raises(PermissionError):
            run(action)


def test_legacy_retain_baseline_remains_nonpromotion():
    result,state=run("continue_lineage_best")
    assert result.records[0].payload["model_promoted"] is False
    assert result.records[0].payload["human_review_still_required"] is False
