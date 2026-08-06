from __future__ import annotations

import json

from hephaestus.data.acquisition_models import (
    AcquiredFileEvidence,
    AcquisitionReceipt,
    TransferAttempt,
)


def _receipt(*, observations: dict[str, str], local_hash: str = "sha256:" + "a" * 64):
    file = AcquiredFileEvidence(
        relative_path="data/train.jsonl",
        source_url="https://provider.invalid/pinned/data/train.jsonl",
        provider_object_id="object-1",
        size_bytes=10,
        provider_declared_hash="a" * 64,
        provider_hash_algorithm="sha256",
        provider_hash_status="verified",
        transport_checksum=None,
        transport_checksum_status="not_provided",
        local_content_hash=local_hash,
        cache_key="cache-key-1",
        cache_status="hit_verified",
        cache_ref="/cache/object-1",
    )
    return AcquisitionReceipt.create(
        plan_id="plan-1",
        selection_decision_id="selection-1",
        approval_refs=("approval://1",),
        candidate_id="candidate-1",
        provider_id="provider-1",
        dataset_id="org/dataset",
        requested_revision="main",
        resolved_revision="b" * 40,
        acquired_files=(file,),
        byte_totals={"acquired": 10, "transferred": 0, "planned": 10},
        cache_status="all_hits",
        dataset_card_ref="https://provider.invalid/pinned/README.md",
        dataset_card_revision="b" * 40,
        license="mit",
        license_source="dataset_card",
        transfer_attempts=(
            TransferAttempt("data/train.jsonl", 1, "cache_reuse", "completed"),
        ),
        partial_recovery_evidence=(),
        artifact_refs=(),
        warnings=(),
        missing_evidence=("transport_checksum:data/train.jsonl",),
        cleanup=(),
        completion_status="completed",
        issues=(),
        observations=observations,
    )


def test_observational_timestamps_do_not_change_receipt_identity() -> None:
    first = _receipt(
        observations={"started_at": "2026-01-01", "completed_at": "2026-01-02"}
    )
    second = _receipt(
        observations={"started_at": "2026-02-01", "completed_at": "2026-02-02"}
    )

    assert first.receipt_id == second.receipt_id
    assert first.deterministic_dict() == second.deterministic_dict()
    assert first.observations != second.observations


def test_material_evidence_changes_receipt_identity_and_round_trip_is_json_safe() -> (
    None
):
    first = _receipt(observations={})
    changed = _receipt(observations={}, local_hash="sha256:" + "c" * 64)

    assert first.receipt_id != changed.receipt_id
    payload = json.loads(json.dumps(first.to_dict()))
    assert AcquisitionReceipt.from_dict(payload).to_dict() == first.to_dict()
