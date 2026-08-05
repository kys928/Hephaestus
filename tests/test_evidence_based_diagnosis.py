from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from hephaestus.diagnosis import (
    EvidenceBasedDiagnosisService,
    MappingEvidenceAdapter,
    StateEvidenceAdapter,
)
from hephaestus.interfaces.services import DiagnosisService
from hephaestus.roles.diagnostician import DiagnosticianRole
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest


def _complete_evidence() -> list[dict[str, object]]:
    return [
        {
            "evidence_kind": "eval_report",
            "source_ref": "eval/report.json",
            "eval_id": "eval-run-1",
            "eval_pack_integrity_level": "content_hash_verified",
            "scorecard_integrity_level": "content_hash_verified",
            "deterministic_scorecard": {"gate_results": {"probe": {"passed": True}}},
        },
        {
            "evidence_kind": "run_record",
            "source_ref": "state/run.json",
            "run_id": "run-1",
            "status": "completed",
        },
        {
            "evidence_kind": "replay_verification",
            "source_ref": "state/replay.json",
            "status": "reproducible",
            "confidence": 0.95,
        },
        {
            "evidence_kind": "dataset_manifest",
            "source_ref": "data/manifest.json",
            "manifest_integrity_level": "complete",
            "tokenizer_compatibility": {"status": "compatible"},
            "wrapper_policy": {"status": "compatible"},
        },
        {
            "evidence_kind": "checkpoint_evidence",
            "source_ref": "checkpoints/integrity.json",
            "checkpoint_verified": True,
        },
        {
            "evidence_kind": "training_metrics",
            "source_ref": "training/metrics.json",
            "optimizer_stable": True,
            "numerically_stable": True,
            "training_sufficient": True,
        },
    ]


def _request(
    *extra: dict[str, object], evidence: list[dict[str, object]] | None = None
) -> DiagnosisRequest:
    records = _complete_evidence() if evidence is None else evidence
    return DiagnosisRequest(
        request_id="diagnose-run-1",
        run_id="run-1",
        lineage_id="lineage-main",
        stage_name="early_pretraining",
        observed_failures=[*records, *extra],
    )


def _leading_domain(report) -> str:
    return next(
        item.failure_domain
        for item in report.hypotheses
        if item.hypothesis_id == report.leading_hypothesis_id
    )


def test_complete_bundle_produces_ranked_valid_diagnosis() -> None:
    service = EvidenceBasedDiagnosisService()
    report = service.diagnose(
        _request(
            {
                "evidence_kind": "data_audit",
                "source_ref": "audit/quality.json",
                "data_quality_failed": True,
            }
        )
    )

    assert report.status == "completed"
    assert _leading_domain(report) == "data_quality"
    assert report.confidence > 0.5
    assert report.missing_evidence == []
    assert report.hypotheses[0].recommended_intervention_kinds
    assert report.metadata["recommendations_executed"] is False
    assert set(report.metadata["epistemic_distinctions"]) == {
        "observation",
        "inference",
        "hypothesis",
        "recommendation",
        "confidence",
    }


def test_inconclusive_when_unknown_evidence_cannot_distinguish_cause() -> None:
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            evidence=[
                {"evidence_kind": "unknown", "source_ref": "unknown:1", "value": 7}
            ]
        )
    )

    assert report.status == "inconclusive"
    assert report.hypotheses[0].failure_domain == "inconclusive"
    assert report.missing_evidence


def test_malformed_evidence_is_reported_without_crash() -> None:
    request = _request()
    request.observed_failures.extend(["bad", 17])  # type: ignore[list-item]

    report = EvidenceBasedDiagnosisService().diagnose(request)

    assert any(issue.category == "invalid_request" for issue in report.issues)
    assert report.observations


def test_contradictory_evidence_lowers_confidence_and_is_visible() -> None:
    baseline = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "data_audit",
                "source_ref": "audit/fail.json",
                "data_quality_failed": True,
            }
        )
    )
    conflicting = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "data_audit",
                "source_ref": "audit/fail.json",
                "data_quality_failed": True,
            },
            {
                "evidence_kind": "data_audit",
                "source_ref": "audit/pass.json",
                "data_quality_verified": True,
            },
        )
    )

    hypothesis = next(
        item for item in conflicting.hypotheses if item.failure_domain == "data_quality"
    )
    assert conflicting.confidence < baseline.confidence
    assert hypothesis.contradicting_evidence_refs == ["audit/pass.json"]


def test_missing_eval_integrity_caps_downstream_confidence() -> None:
    evidence = [
        item for item in _complete_evidence() if item["evidence_kind"] != "eval_report"
    ]
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "tokenizer_check",
                "source_ref": "tokenizer/mismatch.json",
                "tokenizer_mismatch": True,
            },
            evidence=evidence,
        )
    )

    tokenizer = next(
        item for item in report.hypotheses if item.failure_domain == "tokenizer"
    )
    assert tokenizer.confidence <= 0.35
    assert tokenizer.metadata["eval_integrity_confidence_cap_applied"] is True
    assert "verified_eval_integrity" in report.missing_evidence
    assert any(
        issue.code == "eval_integrity_unverified" and issue.blocking
        for issue in report.issues
    )


def test_runtime_incident_takes_precedence_over_data_hypothesis() -> None:
    evidence = [
        item for item in _complete_evidence() if item["evidence_kind"] != "eval_report"
    ]
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "incident_record",
                "source_ref": "runtime/incident.json",
                "severity": "critical",
            },
            {
                "evidence_kind": "data_audit",
                "source_ref": "audit/quality.json",
                "data_quality_failed": True,
            },
            evidence=evidence,
        )
    )

    assert _leading_domain(report) == "runtime_or_system"
    assert report.hypotheses[0].confidence > next(
        item.confidence
        for item in report.hypotheses
        if item.failure_domain == "data_quality"
    )


def test_explicit_wrapper_and_tokenizer_mismatch_are_diagnosed() -> None:
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "wrapper_check",
                "source_ref": "format/wrapper.json",
                "wrapper_compatible": False,
            },
            {
                "evidence_kind": "tokenizer_check",
                "source_ref": "tokenizer/check.json",
                "tokenizer_compatible": False,
            },
        )
    )

    domains = {item.failure_domain for item in report.hypotheses}
    assert "data_format_or_wrapper" in domains
    assert "tokenizer" in domains


def test_numerical_instability_requires_recorded_non_finite_signal() -> None:
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "training_metrics",
                "source_ref": "training/nonfinite.json",
                "non_finite_gradient": True,
            }
        )
    )

    assert _leading_domain(report) == "numerical_instability"
    assert (
        "change_training_recipe" in report.hypotheses[0].recommended_intervention_kinds
    )


@pytest.mark.parametrize(
    ("domain", "signal"),
    [
        ("evaluation_integrity", "eval_integrity_failed"),
        ("launch_or_reproducibility", "launch_config_mismatch"),
        ("data_quality", "malformed_data"),
        ("data_coverage", "data_coverage_gap"),
        ("data_format_or_wrapper", "data_format_mismatch"),
        ("tokenizer", "special_token_mismatch"),
        ("architecture", "architecture_mismatch"),
        ("optimizer_or_scheduler", "scheduler_misconfigured"),
        ("numerical_instability", "numerical_overflow"),
        ("undertraining", "undertraining_detected"),
        ("overfitting", "overfitting_detected"),
        ("decoding", "decoding_artifact"),
        ("runtime_or_system", "hardware_interruption"),
        ("checkpoint_integrity", "checkpoint_hash_mismatch"),
        ("model_family_limitation", "model_family_limitation_detected"),
    ],
)
def test_every_finite_failure_domain_has_a_deterministic_rule(
    domain: str, signal: str
) -> None:
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "explicit_check",
                "source_ref": f"checks/{signal}.json",
                signal: True,
            }
        )
    )

    assert domain in {item.failure_domain for item in report.hypotheses}


def test_confidence_is_bounded_for_all_findings() -> None:
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            {
                "evidence_kind": "training_metrics",
                "source_ref": "training/nonfinite.json",
                "non_finite_loss": True,
                "confidence": 99,
            }
        )
    )

    assert 0.0 <= report.confidence <= 1.0
    assert all(0.0 <= item.confidence <= 1.0 for item in report.observations)
    assert all(0.0 <= item.confidence <= 1.0 for item in report.hypotheses)


def test_identical_input_is_deterministic() -> None:
    service = EvidenceBasedDiagnosisService()
    request = _request(
        {
            "evidence_kind": "data_audit",
            "source_ref": "audit/quality.json",
            "malformed_data": True,
        }
    )

    first = service.diagnose(request).to_dict()
    second = service.diagnose(request).to_dict()

    assert first == second


def test_unknown_record_shape_does_not_crash() -> None:
    report = EvidenceBasedDiagnosisService().diagnose(
        _request(
            evidence=[{"unexpected": {"deep": [object(), {"value": float("inf")}]}}]
        )
    )

    assert report.status == "inconclusive"
    assert report.observations[0].evidence_kind == "unknown"


def test_evidence_adapter_failure_becomes_retryable_issue() -> None:
    class BrokenAdapter:
        def load(self, request: DiagnosisRequest):
            raise RuntimeError("fixture adapter unavailable")

    report = EvidenceBasedDiagnosisService([BrokenAdapter()]).diagnose(_request())

    issue = next(
        item for item in report.issues if item.code == "evidence_adapter_failed"
    )
    assert issue.category == "provider_unavailable"
    assert issue.retryable is True


def test_service_and_role_conform_to_shared_protocol() -> None:
    service = EvidenceBasedDiagnosisService()
    role = DiagnosticianRole(service)

    assert isinstance(service, DiagnosisService)
    assert role.run(_request()).request_id == "diagnose-run-1"


def test_input_and_mapping_adapter_state_are_not_mutated() -> None:
    records = {
        "evidence:quality": {"evidence_kind": "data_audit", "data_quality_failed": True}
    }
    original_records = deepcopy(records)
    request = DiagnosisRequest(
        request_id="diagnose-run-1",
        run_id="run-1",
        lineage_id="lineage-main",
        stage_name="early_pretraining",
        observed_failures=_complete_evidence(),
        evidence_refs=["evidence:quality"],
    )
    original_request = deepcopy(request.to_dict())

    EvidenceBasedDiagnosisService([MappingEvidenceAdapter(records)]).diagnose(request)

    assert request.to_dict() == original_request
    assert records == original_records


def test_state_adapter_reads_fixture_without_mutating_files(tmp_path: Path) -> None:
    run_rows = [
        {
            "run_id": "run-1",
            "lineage_id": "lineage-main",
            "stage_name": "early_pretraining",
            "status": "failed",
        }
    ]
    path = tmp_path / "run_records.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in run_rows),
        encoding="utf-8",
    )
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    request = DiagnosisRequest("req", "run-1", "lineage-main", "early_pretraining")

    report = EvidenceBasedDiagnosisService([StateEvidenceAdapter(tmp_path)]).diagnose(
        request
    )

    after = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    assert before == after
    assert any(item.failure_domain == "runtime_or_system" for item in report.hypotheses)
