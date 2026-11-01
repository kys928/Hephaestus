"""The published large mock fixture must never be eligible for final sampling."""
from pathlib import Path

import pytest

from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.evaluation.public_exposures import public_exposure_catalog


def test_catalog_excludes_every_split_of_the_explicit_large_public_mock_fixture():
    root = Path(__file__).resolve().parents[1]
    result = public_exposure_catalog(root, [])
    mock, _ = generate_original_splits(GeneratedDataSpec(812351, 24, 6, 512, 1000))
    expected = {row["sample_id"] for rows in mock.values() for row in rows}
    assert expected <= set(result["sample_ids"])
    assert result["arithmetic_bucket9_capacity"] == 2021
    assert result["remaining_arithmetic_bucket9_capacity"] < 2007
    assert result["protected_final_material_accessed"] is False


def test_catalog_rejects_a_future_partition_disguised_as_public_pilot():
    root = Path(__file__).resolve().parents[1]
    original, _ = generate_original_splits(GeneratedDataSpec(20261028))
    with pytest.raises(ValueError, match="bucket 8"):
        public_exposure_catalog(root, original["evaluation"][:1])
