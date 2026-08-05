from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hephaestus.data import (
    AutonomousDataPreprocessor,
    DataProcessingConfig,
    DatasetAcquisitionApproval,
    DeterministicDatasetSelectionService,
)
from hephaestus.providers.datasets import LocalFixtureDatasetProvider, LocalFixtureDescriptor
from hephaestus.schemas.discovery_contract import DatasetSearchRequest


@dataclass(slots=True)
class EmailFilter:
    filter_id: str = "test-email-filter-v1"

    def keep(self, record: dict[str, object]) -> bool:
        return "@" not in " ".join(map(str, record.values()))


@dataclass(slots=True)
class NamedContaminationChecker:
    reference_set_id: str = "fixture-eval-set-v1"

    def is_contaminated(self, record: dict[str, object]) -> bool:
        return "benchmark secret" in " ".join(map(str, record.values())).casefold()


@dataclass(slots=True)
class FixtureTokenizerChecker:
    tokenizer_ref: str = "tokenizer-fixture-v1"
    checker_id: str = "whitespace-tokenizer-check-v1"

    def check(self, texts: list[str]) -> tuple[bool, dict[str, object]]:
        return all(text.split() for text in texts), {"records_checked": len(texts)}


def _write_fixture(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"prompt":"Follow this instruction","target":"Return a safe answer"}',
                '{"prompt":"Follow this instruction","target":"Return a safe answer"}',
                "{malformed-json",
                '{"text":"A safe general example"}',
                '{"text":"contact test@example.com"}',
                '{"text":"benchmark secret answer"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _request() -> DatasetSearchRequest:
    return DatasetSearchRequest(
        request_id="data-search-local",
        diagnosis_report_id="diag-local",
        problem_statement="instruction following general capability",
        capability_targets=["instruction_following"],
        required_languages=["en"],
        required_domains=["general"],
        required_formats=["jsonl"],
        tokenizer_ref="tokenizer-fixture-v1",
        license_allowlist=["mit"],
        provider_allowlist=["local_fixture"],
    )


def test_approved_local_fixture_processes_end_to_end_with_honest_evidence(tmp_path: Path) -> None:
    fixture_path = tmp_path / "source.jsonl"
    _write_fixture(fixture_path)
    provider = LocalFixtureDatasetProvider(
        fixtures=(
            LocalFixtureDescriptor(
                path=fixture_path,
                dataset_id="fixture/instructions",
                task_types=("instruction_following",),
                languages=("en",),
                domains=("general",),
                license="mit",
            ),
        )
    )
    candidate = provider.search(_request())[0]
    selection = DeterministicDatasetSelectionService().select(_request(), [candidate])
    assert selection.status == "selected"
    approval = DatasetAcquisitionApproval(
        selection_decision_id=selection.decision_id,
        approved_candidate_ids=(candidate.candidate_id,),
        approval_refs=("approval://operator/data-local-1",),
    )
    processor = AutonomousDataPreprocessor(
        DataProcessingConfig(artifact_root=tmp_path / "artifacts", chunk_size_tokens=64),
        record_filter=EmailFilter(),
        contamination_checker=NamedContaminationChecker(),
        tokenizer_checker=FixtureTokenizerChecker(),
    )

    first = processor.process(
        run_id="run-data-1",
        lineage_id="lineage-main",
        stage_name="targeted_repair",
        candidate=candidate,
        selection=selection,
        approval=approval,
        tokenizer_ref="tokenizer-fixture-v1",
    )
    second = processor.process(
        run_id="run-data-1",
        lineage_id="lineage-main",
        stage_name="targeted_repair",
        candidate=candidate,
        selection=selection,
        approval=approval,
        tokenizer_ref="tokenizer-fixture-v1",
    )

    evidence = first.processing_evidence
    assert first.processed_content_hash == second.processed_content_hash
    assert first.artifact_dir == second.artifact_dir
    assert evidence["audit_scope"] == "full_scan"
    assert evidence["sample_validation"]["malformed_rows"] == 1
    assert evidence["deduplication"]["exact_duplicates_removed"] == 1
    assert evidence["deduplication"]["approximate_status"] == "approximate_deduplication_not_run"
    assert evidence["filtering"]["pii_integrity_claim"] == "filter_applied_not_proven_complete"
    assert evidence["contamination"]["status"] == "checked_against_named_reference_set"
    assert evidence["contamination"]["reference_set_id"] == "fixture-eval-set-v1"
    assert evidence["tokenizer_compatibility"]["status"] == "checked"
    assert first.manifest.manifest_integrity_level == "complete"
    assert first.preprocessing_report.dropped_examples == 4
    assert Path(first.trainable_data_contract.processed_dataset_ref).is_file()
    assert (first.artifact_dir / "dataset_manifest.json").is_file()
    assert (first.artifact_dir / "processing_evidence.json").is_file()


def test_acquisition_rejects_missing_explicit_approval(tmp_path: Path) -> None:
    fixture_path = tmp_path / "source.jsonl"
    _write_fixture(fixture_path)
    provider = LocalFixtureDatasetProvider(
        fixtures=(LocalFixtureDescriptor(path=fixture_path, dataset_id="fixture", license="mit"),)
    )
    request = DatasetSearchRequest(
        request_id="search",
        diagnosis_report_id="diag",
        problem_statement="fixture",
        license_allowlist=["mit"],
        provider_allowlist=["local_fixture"],
    )
    candidate = provider.search(request)[0]
    selection = DeterministicDatasetSelectionService(minimum_score=0.0).select(request, [candidate])
    approval = DatasetAcquisitionApproval(selection.decision_id, (), ())

    with pytest.raises(PermissionError, match="approval"):
        AutonomousDataPreprocessor(DataProcessingConfig(artifact_root=tmp_path / "artifacts")).process(
            run_id="run",
            lineage_id="lineage",
            stage_name="smoke_test",
            candidate=candidate,
            selection=selection,
            approval=approval,
        )


def test_unchecked_integrity_levels_are_not_overstated(tmp_path: Path) -> None:
    fixture_path = tmp_path / "source.jsonl"
    fixture_path.write_text('{"text":"one safe record"}\n', encoding="utf-8")
    provider = LocalFixtureDatasetProvider(
        fixtures=(LocalFixtureDescriptor(path=fixture_path, dataset_id="fixture", license="mit"),)
    )
    request = DatasetSearchRequest("search", "diag", "fixture", license_allowlist=["mit"])
    candidate = provider.search(request)[0]
    selection = DeterministicDatasetSelectionService(minimum_score=0.0).select(request, [candidate])
    approval = DatasetAcquisitionApproval(selection.decision_id, (candidate.candidate_id,), ("approval://1",))

    result = AutonomousDataPreprocessor(DataProcessingConfig(artifact_root=tmp_path / "artifacts")).process(
        run_id="run",
        lineage_id="lineage",
        stage_name="smoke_test",
        candidate=candidate,
        selection=selection,
        approval=approval,
        tokenizer_ref="declared-tokenizer",
    )

    assert result.processing_evidence["filtering"]["status"] == "not_checked"
    assert result.processing_evidence["contamination"]["status"] == "not_checked"
    assert result.processing_evidence["tokenizer_compatibility"]["status"] == "declared_not_verified"
