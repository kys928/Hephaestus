from hashlib import sha256

import pytest

from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.schemas.research_objective import ResearchAdmissionError
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def test_production_intake_anchors_reviewed_scope_in_existing_store(tmp_path):
    runtime = ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path / "state", artifact_root=tmp_path / "evidence",
    )).build()
    catalog = tmp_path / "catalog.json"
    raw = canonical_bytes({
        "schema": "research-source-catalog.v1",
        "sources": [
            {
                "source_id": role,
                "subject": role,
                "study_scope": "wrapper-supervision",
                "source_uri": f"runpod-s3://volume/hephaestus/scientific/v1/{role}.jsonl",
                "sha256_ref": "sha256:" + digest * 64,
                "expected_rows": 4,
                "expected_bytes": 100,
                "role": role,
                "capability": "diagnosis_coverage_v1",
            }
            for role, digest in (("original", "0"), ("candidate", "1"))
        ],
    })
    catalog.write_bytes(raw)
    catalog_ref = "sha256:" + sha256(raw).hexdigest()
    kwargs = dict(run_id="research-a", lineage_id="lineage-a", reviewed_catalogs=((catalog, catalog_ref),))
    question = "Investigate Diagnostician coverage of wrapper supervision"
    ref = runtime.intake_research_objective(question, **kwargs)
    assert ref == runtime.intake_research_objective(question, **kwargs)
    report = read_verified_json(runtime.artifact_store, ref)
    assert report["source_refs"] == ["sha256:" + c * 64 for c in "01"]
    assert report["reviewed_catalog_ref"] == catalog_ref
    assert report["execution_authorized"] is False
    assert runtime.research_compute_ledger().repository is runtime.state_repository
    assert runtime.research_compute_ledger().artifacts is runtime.artifact_store
    with pytest.raises(ResearchAdmissionError):
        runtime.intake_research_objective(question, required_scope="unreviewed", **kwargs)
    with pytest.raises(ResearchAdmissionError):
        runtime.intake_research_objective("Train a replacement candidate immediately", **kwargs)
    catalog.write_bytes(raw + b"\n")
    with pytest.raises(ResearchAdmissionError, match="independently pinned"):
        runtime.intake_research_objective(question, **kwargs)
