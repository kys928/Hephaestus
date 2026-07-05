"""Build schema-backed trainable data contracts."""

from __future__ import annotations

from hephaestus.data.chunking import chunking_profile
from hephaestus.schemas.trainable_data_contract import TrainableDataContract


def build_trainable_data_contract(
    *,
    run_id: str,
    manifest_id: str,
    processed: dict[str, object],
    processed_dataset_ref: str,
) -> TrainableDataContract:
    chunk_profile = chunking_profile(processed)
    return TrainableDataContract(
        contract_id=str(processed.get("contract_id") or f"contract-{run_id}"),
        run_id=run_id,
        manifest_id=manifest_id,
        processed_dataset_ref=processed_dataset_ref,
        schema_version=str(processed.get("schema_version") or "v1"),
        min_tokens=int(processed.get("min_tokens") or chunk_profile.get("min_tokens") or 256),
    )
