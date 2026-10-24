"""Model-originated decisions affect real bounded plans; no GPU or training."""
from dataclasses import replace
import json
import pytest

from hephaestus.planning.service import ClosedLoopExperimentPlanner, ExperimentPlanningError
from hephaestus.schemas.diagnosis_contract import DiagnosticHypothesis, DiagnosisRequest
from hephaestus.schemas.model_proposal import ModelInterventionSelection
from hephaestus.policy.model_proposals import PLANNER_DECISIONS, planner_selection
from hephaestus.llm.role_contract import RoleAnswer, RoleMasteryContract
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.state import RepositoryIntegrationRecordSink
from hephaestus.control.staged_state import WORKFLOW_RECORD_COLLECTION
from hephaestus.control.autonomous_experiment import GuardedTrainingLifecycleService
from test_model_backed_staged_roles import _run, ModelProposals
from test_closed_loop_planner import _diagnosis
from test_certified_role_inference import CONTRACT


class SelectRepairEvaluation(ModelProposals):
    def generate(self, **kwargs):
        result = super().generate(**kwargs)
        if kwargs["spec"].role == "planner":
            raw = json.loads(result.text)
            raw.update(decision="repair_evaluation", action="request_recheck", primary_variable="evaluation_protocol")
            return replace(result, text=json.dumps(raw))
        return result


def test_changing_model_selection_changes_the_downstream_experiment(tmp_path):
    first, repo1, _, _ = _run(tmp_path / "first", ModelProposals())
    second, repo2, _, _ = _run(tmp_path / "second", SelectRepairEvaluation())
    assert first.status == second.status == "completed"
    plans = [[r["payload"] for r in repo.all(WORKFLOW_RECORD_COLLECTION) if r["kind"] == "experiment_proposal"][0]
             for repo in (repo1, repo2)]
    assert [p["primary_variable"] for p in plans] == ["diagnostic_measurement", "evaluation_protocol"]
    assert plans[0]["experiment_id"] != plans[1]["experiment_id"]
    for plan in plans:
        assert plan["metadata"]["model_originated"] is True
        assert plan["metadata"]["model_inference_receipt_ref"].startswith("sha256:")
        assert plan["training_constraints"]["training_required"] is False


def test_invalid_controller_model_cannot_execute_its_delegate(tmp_path, monkeypatch):
    from test_staged_autonomous_orchestration import FakeOperationRouter
    calls = []
    original = FakeOperationRouter.execute
    def track(self, request):
        calls.append(request.substep)
        return original(self, request)
    monkeypatch.setattr(FakeOperationRouter, "execute", track)
    result, repo, _, _ = _run(tmp_path, ModelProposals(corrupt_role="controller"))
    assert result.status == "terminal_failure"
    assert calls == []
    assert {r["kind"] for r in repo.all(WORKFLOW_RECORD_COLLECTION)} <= {"subsystem_failure", "substep_result"}


def _duration_diagnosis():
    return _diagnosis(hypotheses=[DiagnosticHypothesis("duration", "undertraining", "Healthy but insufficient training.",
        supporting_evidence_refs=["evidence://duration"], recommended_intervention_kinds=["resume_training", "collect_more_evidence"],
        confidence=.9)])


def _selection():
    return ModelInterventionSelection("resume_training", "training_duration", .8,
        ["evidence://duration"], "sha256:"+"b"*64, "Run a bounded continuation.")


def test_admitted_training_plan_changes_duration_and_preserves_checkpoint_and_approvals():
    planner, diagnosis = ClosedLoopExperimentPlanner(), _duration_diagnosis()
    chosen = planner.admit_model_selection(diagnosis, _selection(), {"evidence://duration"})
    proposal = planner.propose_experiment(diagnosis, chosen, None, None)
    assert proposal.primary_variable == "training_duration"
    assert proposal.baseline_ref == diagnosis.metadata["baseline_ref"]
    assert proposal.training_constraints["one_primary_variable"] == "training_duration"
    assert proposal.training_constraints["training_required"] is True
    assert proposal.metadata["model_inference_receipt_ref"] == _selection().inference_receipt_ref
    assert set(chosen.metadata["required_approvals"]) <= set(proposal.required_approvals)


@pytest.mark.parametrize("fault", ["budget", "scope", "evidence", "kind", "primary"])
def test_invalid_model_selection_is_not_admitted(fault):
    planner, diagnosis, selection = ClosedLoopExperimentPlanner(), _duration_diagnosis(), _selection()
    if fault == "budget": diagnosis.metadata["budget"] = {"max_compute_hours":0}
    elif fault == "scope": diagnosis.metadata["allowed_intervention_kinds"] = ["collect_more_evidence"]
    elif fault == "evidence": selection.evidence_refs = ["fabricated://evidence"]
    elif fault == "kind": selection.intervention_kind = "change_tokenizer"
    else: selection.primary_variable = "model_candidate"
    with pytest.raises(ExperimentPlanningError):
        planner.admit_model_selection(diagnosis, selection, {"evidence://duration"})


def test_every_frozen_planner_decision_has_an_explicit_translation():
    contract = RoleMasteryContract.load(CONTRACT)
    assert set(PLANNER_DECISIONS) == set(contract.profile("planner")["vocabulary"]["decision"])
    for decision, (kind, action, variables) in PLANNER_DECISIONS.items():
        for model_variable, primary in variables.items():
            answer = RoleAnswer(decision, action, model_variable, .7, ("evidence://1",), (), "Bounded selection.")
            result = planner_selection(answer, "sha256:"+"a"*64)
            assert (result.intervention_kind, result.primary_variable) == (kind, primary)
            bad = replace(answer, action="abort_run" if action != "abort_run" else "request_recheck")
            with pytest.raises(ValueError): planner_selection(bad, "sha256:"+"a"*64)


def test_resumed_production_campaign_retrieves_failed_intervention_and_avoids_it(tmp_path):
    settings = ProductionCompositionSettings(tmp_path / "state", tmp_path / "artifacts")
    runtime = ProductionCompositionRoot(settings).build()
    diagnosis = _duration_diagnosis()
    selected = runtime.planner.admit_model_selection(diagnosis, _selection(), {"evidence://duration"})
    proposal = runtime.planner.propose_experiment(diagnosis, selected, None, None)
    sink = runtime.coordinator.record_sink
    sink.append("intervention_proposal", selected.to_dict())
    sink.append("experiment_proposal", proposal.to_dict())
    sink.append("experiment_comparison", {"comparison_id":"comparison-failed", "experiment_id":proposal.experiment_id,
        "primary_outcome":"regressed", "deterministic_gate_status":"failed"})
    restarted = ProductionCompositionRoot(settings).build()
    history = restarted.planner.memory_query.intervention_history(diagnosis.lineage_id)
    assert history and history[0]["metadata"]["intervention_kind"] == "resume_training"
    with pytest.raises(ExperimentPlanningError, match="not_scientifically_admissible"):
        restarted.planner.admit_model_selection(diagnosis, _selection(), {"evidence://duration"})
    assert all(p.intervention_kind != "resume_training" for p in restarted.planner.propose_interventions(diagnosis))
    # A genuinely new supporting observation may justify another controlled test.
    diagnosis.hypotheses[0].supporting_evidence_refs.append("evidence://new-duration-probe")
    retry = restarted.planner.admit_model_selection(diagnosis, _selection(), {"evidence://duration"})
    assert retry.metadata["new_evidence_overrides_dead_end"] is True


def test_unresolved_discovery_design_cannot_launch_even_if_status_is_forged_ready():
    planner = ClosedLoopExperimentPlanner()
    diagnosis = _diagnosis()
    intervention = next(p for p in planner.propose_interventions(diagnosis) if p.intervention_kind == "replace_or_mix_dataset")
    proposal = planner.propose_experiment(diagnosis, intervention, None, None, allow_unresolved_discovery=True)
    assert proposal.metadata["unresolved_discovery"] == ["dataset_selection"]
    class NoTraining:
        def launch(self, proposal):
            pytest.fail("unresolved experimental design reached a training backend")
    result = GuardedTrainingLifecycleService(NoTraining()).launch(replace(proposal, status="approved"))
    assert result.status == "failed" and result.metadata["launch_attempted"] is False


def test_composition_diagnosis_resolves_bound_previous_run_evidence(tmp_path):
    from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
    from hephaestus.storage.verified import canonical_bytes
    runtime = ProductionCompositionRoot(ProductionCompositionSettings(tmp_path / "state", tmp_path / "artifacts")).build()
    doc = {"run_id":"previous", "lineage_id":"L", "stage_name":"smoke_test", "kind":"eval_report",
           "payload":{"eval_integrity_verified":True, "deterministic_scorecard":{"passed":True},
                      "checkpoint_verified":True, "training_sufficient":False}}
    artifact = runtime.artifact_store.put_bytes(canonical_bytes(doc))
    runtime.model_evidence.store.register(ScientificEvidenceBinding("bound", "current", "previous", "L",
        "smoke_test", "eval_report", artifact.artifact_ref))
    restarted = ProductionCompositionRoot(runtime.settings).build()
    report = restarted.diagnosis_service.diagnose(DiagnosisRequest("req", "current", "L", "smoke_test",
        evidence_refs=[artifact.artifact_ref]))
    assert artifact.artifact_ref in [o.source_ref for o in report.observations]
    assert len(restarted.diagnosis_service.delegate.evidence_adapters) == 2


@pytest.mark.parametrize("fault", ["superseded", "rebound_model_decision"])
def test_selected_plan_revalidates_source_freshness_and_raw_decision_before_use(tmp_path, fault):
    from hephaestus.production.model_roles import ModelBackedStage
    from hephaestus.production.evidence import ArtifactModelEvidenceBinder
    from hephaestus.state.evidence_store import ScientificEvidenceStore, EVIDENCE_BINDINGS
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    from hephaestus.storage.verified import canonical_bytes, read_verified_json
    from hephaestus.control.staged_state import StagedOperationRequest
    state, repo, model, s3 = _run(tmp_path, ModelProposals())
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    binder = ArtifactModelEvidenceBinder(ScientificEvidenceStore(repo, artifacts))
    prior = dict(state.steps["planner.intervention_ranking"].output)
    request = StagedOperationRequest(state.workflow_id, state.run_id, state.lineage_id, state.stage_name,
                                    "planner", "experiment_proposal", "op-replay", 2, (), {"planner.intervention_ranking":prior})
    document = read_verified_json(artifacts, prior["selected_intervention_ref"])
    if fault == "superseded":
        ref = document["selection"]["evidence_refs"][0]
        source = read_verified_json(artifacts, ref)
        replacement = artifacts.put_bytes(canonical_bytes({**source, "payload":{**source["payload"], "new_observation":True}}))
        binder.store.supersede(ref, replacement.artifact_ref, run_id=state.run_id, lineage_id=state.lineage_id)
    else:
        from hephaestus.schemas.model_proposal import ModelInterventionSelection
        changed = {**document["selection"], "intervention_kind":"repair_evaluation", "primary_variable":"evaluation_protocol"}
        selected = ClosedLoopExperimentPlanner().admit_model_selection(binder.diagnosis_for(request),
            ModelInterventionSelection.from_dict(changed), set(changed["evidence_refs"]))
        forged = artifacts.put_bytes(canonical_bytes({"selection":changed,"intervention":selected.to_dict()}))
        prior["selected_intervention_ref"] = forged.artifact_ref
    class ReceiptReader:
        artifact_store = artifacts
        contract = RoleMasteryContract.load(CONTRACT)
    class NoDelegate:
        def execute(self, request): pytest.fail("invalid selected plan reached delegate")
    stage = ModelBackedStage("planner", NoDelegate(), ReceiptReader(), binder, ClosedLoopExperimentPlanner())
    with pytest.raises(ValueError):
        stage.execute(request)
