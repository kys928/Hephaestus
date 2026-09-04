from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from hephaestus.data import (
    AutonomousDataPreprocessor,
    DataProcessingConfig,
    DatasetAcquisitionApproval,
)
from hephaestus.data.acquisition_models import AcquisitionReceipt, AcquiredFileEvidence
from hephaestus.data.materialization import validate_remote_acquisition_for_preprocessing
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision


REVISION = "a" * 40
APPROVAL_REF = "approval://operator/remote-preprocessing"


def _candidate() -> DatasetCandidate:
    return DatasetCandidate(
        candidate_id="dataset-hf-real-1",
        provider_id="huggingface",
        dataset_id="org/real-corpus",
        revision=REVISION,
        splits=["train"],
        languages=["en"],
        domains=["general_text"],
        format_profile={"record_format": "parquet"},
        license="cc-by-sa-3.0",
        provenance={"kind": "hub_metadata", "sha": REVISION},
        trust_level="external_metadata",
        compatibility={"remote_code_required": False},
        evidence_refs=["https://huggingface.co/datasets/org/real-corpus"],
    )


def _selection(candidate: DatasetCandidate) -> DatasetSelectionDecision:
    return DatasetSelectionDecision(
        decision_id="dataset-selection-real-1",
        request_id="dataset-search-real-1",
        status="selected",
        selected_candidate_ids=[candidate.candidate_id],
        ranked_candidate_ids=[candidate.candidate_id],
        mixture_weights={candidate.candidate_id: 1.0},
    )


def _approval(
    candidate: DatasetCandidate, selection: DatasetSelectionDecision
) -> DatasetAcquisitionApproval:
    return DatasetAcquisitionApproval(
        selection_decision_id=selection.decision_id,
        approved_candidate_ids=(candidate.candidate_id,),
        approval_refs=(APPROVAL_REF,),
    )


def _receipt(
    source: Path,
    candidate: DatasetCandidate,
    selection: DatasetSelectionDecision,
    *,
    relative_path: str = "data/train.jsonl",
    local_hash: str | None = None,
) -> AcquisitionReceipt:
    digest = local_hash or hashlib.sha256(source.read_bytes()).hexdigest()
    acquired = AcquiredFileEvidence(
        relative_path=relative_path,
        source_url=(
            "https://huggingface.co/datasets/org/real-corpus/resolve/"
            f"{REVISION}/{relative_path}"
        ),
        provider_object_id="provider-object-1",
        size_bytes=source.stat().st_size,
        provider_declared_hash=digest,
        provider_hash_algorithm="sha256",
        provider_hash_status="verified",
        transport_checksum=digest,
        transport_checksum_status="verified",
        local_content_hash=f"sha256:{digest}",
        cache_key="cache-key-1",
        cache_status="stored",
        cache_ref=str(source),
        artifact_ref=f"sha256:{digest}",
        artifact_store_content_hash=digest,
    )
    return AcquisitionReceipt.create(
        plan_id="dataset-acquisition-plan-real-1",
        selection_decision_id=selection.decision_id,
        approval_refs=(APPROVAL_REF,),
        candidate_id=candidate.candidate_id,
        provider_id=candidate.provider_id,
        dataset_id=candidate.dataset_id,
        requested_revision=REVISION,
        resolved_revision=REVISION,
        acquired_files=(acquired,),
        byte_totals={"acquired": source.stat().st_size},
        cache_status="misses_stored",
        dataset_card_ref=(
            "https://huggingface.co/datasets/org/real-corpus/blob/"
            f"{REVISION}/README.md"
        ),
        dataset_card_revision=REVISION,
        license=candidate.license,
        license_source="huggingface_dataset_card",
        transfer_attempts=(),
        partial_recovery_evidence=(),
        artifact_refs=(f"sha256:{digest}",),
        warnings=(),
        missing_evidence=(),
        cleanup=(),
        completion_status="completed",
        issues=(),
    )


def test_completed_remote_receipt_enters_existing_preprocessing_without_provenance_relabel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"text":"A real sentence for model training."}\n'
        '{"text":"A second real sentence with distinct content."}\n',
        encoding="utf-8",
    )
    candidate = _candidate()
    selection = _selection(candidate)
    approval = _approval(candidate, selection)
    receipt = _receipt(source, candidate, selection)

    result = AutonomousDataPreprocessor(
        DataProcessingConfig(
            artifact_root=tmp_path / "artifacts",
            max_input_bytes=1024 * 1024,
            max_rows=100,
            chunk_size_tokens=64,
        )
    ).process_remote_acquisition(
        run_id="first-real-data",
        lineage_id="lineage-first-scientific",
        stage_name="smoke_test",
        candidate=candidate,
        selection=selection,
        approval=approval,
        receipt=receipt,
    )

    assert result.manifest.metadata["provider_id"] == "huggingface"
    assert result.manifest.datasets[0]["source"] == receipt.acquired_files[0].local_content_hash
    assert result.manifest.datasets[0]["version"] == REVISION
    assert result.manifest.datasets[0]["license"] == "cc-by-sa-3.0"
    assert result.processing_evidence["source_acquisition"]["receipt_id"] == receipt.receipt_id
    assert result.processing_evidence["source_acquisition"]["provider_id"] == "huggingface"
    assert result.trainable_data_contract.processed_dataset_ref.endswith("trainable.jsonl")
    assert Path(result.trainable_data_contract.processed_dataset_ref).is_file()


def test_remote_materialization_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"original"}\n', encoding="utf-8")
    candidate = _candidate()
    selection = _selection(candidate)
    approval = _approval(candidate, selection)
    receipt = _receipt(source, candidate, selection)
    source.write_text('{"text":"mutated after acquisition"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="byte size|sha256"):
        validate_remote_acquisition_for_preprocessing(
            candidate=candidate,
            selection=selection,
            approval=approval,
            receipt=receipt,
            max_bytes=1024 * 1024,
        )


def test_remote_materialization_rejects_candidate_revision_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"content"}\n', encoding="utf-8")
    candidate = _candidate()
    selection = _selection(candidate)
    approval = _approval(candidate, selection)
    receipt = _receipt(source, candidate, selection)
    candidate.revision = "b" * 40

    with pytest.raises(ValueError, match="revision"):
        validate_remote_acquisition_for_preprocessing(
            candidate=candidate,
            selection=selection,
            approval=approval,
            receipt=receipt,
            max_bytes=1024 * 1024,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow is an optional Parquet preprocessing dependency",
)
def test_remote_parquet_is_bounded_and_preserves_provider_evidence(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "train.parquet"
    pq.write_table(
        pa.table({"text": ["row one", "row two", "row three"]}),
        source,
    )
    candidate = _candidate()
    selection = _selection(candidate)
    approval = _approval(candidate, selection)
    receipt = _receipt(
        source,
        candidate,
        selection,
        relative_path="data/train.parquet",
    )

    result = AutonomousDataPreprocessor(
        DataProcessingConfig(
            artifact_root=tmp_path / "artifacts",
            max_input_bytes=1024 * 1024,
            max_rows=2,
            chunk_size_tokens=64,
        )
    ).process_remote_acquisition(
        run_id="parquet-real-data",
        lineage_id="lineage-first-scientific",
        stage_name="smoke_test",
        candidate=candidate,
        selection=selection,
        approval=approval,
        receipt=receipt,
    )

    assert result.processing_evidence["sample_validation"]["truncated_at_max_rows"] is True
    assert result.processing_evidence["source_acquisition"]["provider_id"] == "huggingface"
    assert "parquet_record_decode" in result.preprocessing_report.operations
