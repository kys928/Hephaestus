from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.diagnosis_contract import DiagnosisReport
from hephaestus.schemas.discovery_contract import DatasetSelectionDecision, ModelSelectionDecision
from hephaestus.schemas.experiment_contract import ExperimentComparison, TrainingRunHandle
from hephaestus.schemas.lifecycle_contract import LifecycleTransition


def test_diagnosis_report_round_trip_preserves_nested_contracts() -> None:
    report = DiagnosisReport.from_dict({
        "report_id": "diag-1", "request_id": "req-1", "run_id": "run-1",
        "lineage_id": "lin-1", "stage_name": "smoke_test", "status": "inconclusive",
        "observations": [{"observation_id": "obs-1", "evidence_kind": "metric", "source_ref": "metrics.json", "summary": "loss diverged", "confidence": 1.7}],
        "hypotheses": [{"hypothesis_id": "hyp-1", "failure_domain": "not-valid", "summary": "unknown", "confidence": -1}],
        "issues": [{"code": "missing", "category": "missing_evidence", "message": "need gradients"}],
    })
    restored = DiagnosisReport.from_dict(report.to_dict())
    assert restored.observations[0].confidence == 1.0
    assert restored.hypotheses[0].failure_domain == "inconclusive"
    assert restored.issues[0].category == "missing_evidence"


def test_selection_decisions_normalize_status_and_confidence() -> None:
    dataset = DatasetSelectionDecision.from_dict({"decision_id": "d1", "request_id": "r1", "status": "invalid", "confidence": 5})
    model = ModelSelectionDecision.from_dict({"decision_id": "m1", "request_id": "r2", "status": "selected", "confidence": 0.5})
    assert dataset.status == "inconclusive" and dataset.confidence == 1.0
    assert model.status == "selected" and model.confidence == 0.5


def test_training_and_comparison_round_trip_issues() -> None:
    issue = ContractIssue("runtime", "runtime_failure", "process exited", retryable=True)
    handle = TrainingRunHandle.from_dict({"run_id": "run", "experiment_id": "exp", "backend_id": "local", "status": "failed", "issues": [issue.to_dict()]})
    comparison = ExperimentComparison.from_dict({"comparison_id": "cmp", "experiment_id": "exp", "baseline_run_id": None, "issues": [issue.to_dict()], "confidence": 0.2})
    assert handle.issues[0].retryable is True
    assert comparison.issues[0].category == "runtime_failure"


def test_lifecycle_transition_accepts_only_documented_edges() -> None:
    allowed = LifecycleTransition("t1", "experiment", "e1", "training", "evaluation_pending", "run_completed", "runtime_monitor", "completed")
    blocked = LifecycleTransition("t2", "experiment", "e1", "diagnosis_pending", "training", "skip", "planner", "unsafe")
    assert allowed.is_allowed() is True
    assert blocked.is_allowed() is False
