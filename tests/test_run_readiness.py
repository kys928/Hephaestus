from pathlib import Path

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.policy.run_readiness import RunReadinessPolicy
from hephaestus.schemas.stage_contract import StageContract
from hephaestus.state.report_store import ReportStore


def _contract(**overrides: object) -> StageContract:
    payload = {
        "contract_id": "stage-contract-test",
        "stage_name": "early_pretraining",
        "eval_pack_ref": "pretraining_probes",
        "allowed_backends": ["dry_run", "local_process"],
        "required_manifest_fields": ["manifest_id", "run_id", "lineage_id", "datasets", "mixture_weights"],
        "required_data_contract_fields": [
            "contract_id",
            "run_id",
            "manifest_id",
            "processed_dataset_ref",
            "schema_version",
            "min_tokens",
        ],
        "required_contract_refs": [
            "stage.eval_pack_ref",
            "manifest.stage_data_policy_ref",
            "manifest.tokenizer_ref",
        ],
        "accepted_eval_pack_integrity_levels": ["content_hash_verified", "reference_only", "inline_unhashed"],
        "min_manifest_completeness": 0.45,
    }
    payload.update(overrides)
    return StageContract(**payload)


def _manifest(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_id": "manifest-r1",
        "run_id": "r1",
        "lineage_id": "lineage-main",
        "datasets": [{"dataset_id": "d1", "row_count": 10, "version": "v1"}],
        "mixture_weights": {"d1": 1.0},
        "manifest_integrity_level": "complete",
        "completeness_score": 1.0,
        "stage_data_policy_ref": "policy://data/early",
        "tokenizer_ref": "tokenizer://demo",
    }
    payload.update(overrides)
    return payload


def _data_contract(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_id": "contract-r1",
        "run_id": "r1",
        "manifest_id": "manifest-r1",
        "processed_dataset_ref": "artifacts/r1/processed.jsonl",
        "schema_version": "v1",
        "min_tokens": 256,
    }
    payload.update(overrides)
    return payload


def _eval_pack(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "eval_pack_id": "pack-1",
        "eval_pack_version": "v1",
        "eval_pack_integrity_level": "reference_only",
    }
    payload.update(overrides)
    return payload


def test_readiness_blocks_missing_manifest_fields() -> None:
    report = RunReadinessPolicy().evaluate(
        run_id="r1",
        lineage_id="lineage-main",
        stage_name="early_pretraining",
        stage_contract=_contract(),
        backend_name="dry_run",
        dataset_manifest=_manifest(datasets=[], completeness_score=0.2),
        data_contract=_data_contract(),
        eval_pack=_eval_pack(),
    )

    assert report.status == "blocked"
    assert report.launch_allowed is False
    assert any(item.startswith("missing_manifest_fields") for item in report.blockers)
    assert "manifest_completeness_below_stage_minimum" in report.blockers


def test_readiness_blocks_unsupported_backend_stage_pair() -> None:
    report = RunReadinessPolicy().evaluate(
        run_id="r1",
        lineage_id="lineage-main",
        stage_name="early_pretraining",
        stage_contract=_contract(allowed_backends=["dry_run"]),
        backend_name="ardor",
        dataset_manifest=_manifest(),
        data_contract=_data_contract(),
        eval_pack=_eval_pack(),
    )

    assert report.status == "blocked"
    assert "unsupported_backend:ardor" in report.blockers
    assert report.checks["backend"]["passed"] is False


def test_readiness_marks_weak_eval_pack_integrity_inconclusive_without_blocking() -> None:
    report = RunReadinessPolicy().evaluate(
        run_id="r1",
        lineage_id="lineage-main",
        stage_name="early_pretraining",
        stage_contract=_contract(),
        backend_name="dry_run",
        dataset_manifest=_manifest(),
        data_contract=_data_contract(),
        eval_pack=_eval_pack(eval_pack_integrity_level="inline_unhashed"),
    )

    assert report.status == "inconclusive"
    assert report.launch_allowed is True
    assert "weak_eval_pack_integrity:inline_unhashed" in report.warnings
    assert report.checks["eval_pack"]["strong_integrity"] is False


def test_readiness_records_missing_contract_refs() -> None:
    report = RunReadinessPolicy().evaluate(
        run_id="r1",
        lineage_id="lineage-main",
        stage_name="early_pretraining",
        stage_contract=_contract(eval_pack_ref=""),
        backend_name="dry_run",
        dataset_manifest=_manifest(stage_data_policy_ref=None, tokenizer_ref=None),
        data_contract=_data_contract(),
        eval_pack=_eval_pack(),
    )

    assert report.status == "inconclusive"
    assert report.launch_allowed is True
    assert any(item.startswith("missing_contract_refs:") for item in report.warnings)
    assert set(report.checks["contract_refs"]["missing_refs"]) == {
        "stage.eval_pack_ref",
        "manifest.stage_data_policy_ref",
        "manifest.tokenizer_ref",
    }


def test_dry_run_orchestrator_persists_readiness_report(tmp_path: Path) -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id="readiness-dry")
    results = orch.run("readiness-dry")
    outputs = {result.phase.value: result.output for result in results}

    readiness = outputs["training_engineer"]["run_readiness_report"]
    reports = [row for row in ReportStore(tmp_path).all() if row.get("kind") == "run_readiness_report"]

    assert readiness["status"] == "inconclusive"
    assert readiness["launch_allowed"] is True
    assert reports[-1]["report_id"] == "readiness-readiness-dry"
