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


def test_production_evaluation_only_reopens_immutable_intake_and_verifies_source_bytes(tmp_path):
    from io import BytesIO

    from hephaestus.research.source_evidence import ReadOnlyS3SourceReader

    runtime = ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path / "state", artifact_root=tmp_path / "evidence",
    )).build()
    root = "hephaestus/scientific/v1/"
    contents = {
        root + "original.jsonl": canonical_bytes({"text": "plain prose"}) + b"\n",
        root + "candidate.jsonl": canonical_bytes({
            "text": "<|prompt|>\nWhat is Mars?\n<|target|>\nA planet",
        }) + b"\n",
    }

    class FixtureGetOnly:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "test-volume"
            return {"Body": BytesIO(contents[Key])}

    catalog = tmp_path / "catalog.json"
    raw = canonical_bytes({
        "schema": "research-source-catalog.v1",
        "sources": [
            {
                "source_id": role,
                "subject": role,
                "study_scope": "coverage-probe",
                "source_uri": "runpod-s3://test-volume/" + key,
                "sha256_ref": "sha256:" + sha256(contents[key]).hexdigest(),
                "expected_rows": 1,
                "expected_bytes": len(contents[key]),
                "role": role,
                "capability": "diagnosis_coverage_v1",
            }
            for role in ("original", "candidate")
            for key in (root + role + ".jsonl",)
        ],
    })
    catalog.write_bytes(raw)
    catalog_ref = "sha256:" + sha256(raw).hexdigest()
    eval_ref = runtime.artifact_store.put_bytes(canonical_bytes({
        "generation_probes": [{"prompt": "What is Mars?"}],
    })).artifact_ref
    ids = {"run_id": "research-a", "lineage_id": "lineage-a"}
    intake = runtime.intake_research_objective(
        "Investigate Diagnostician coverage of wrapper supervision",
        reviewed_catalogs=((catalog, catalog_ref),), **ids,
    )
    args = dict(
        catalog_path=catalog, independently_pinned_catalog_ref=catalog_ref, eval_pack_ref=eval_ref,
        source_reader=ReadOnlyS3SourceReader(FixtureGetOnly(), "test-volume"),
    )
    evidence = runtime.observe_reviewed_research_sources(intake, **ids, **args)
    assert evidence == runtime.observe_reviewed_research_sources(intake, **ids, **args)
    report = read_verified_json(runtime.artifact_store, evidence)
    assert report["eval_pack_ref"] == eval_ref
    assert report["model_inference_performed"] is False
    assert len(report["observation_refs"]) == 2
    assert len(runtime.state_repository.all("research_source_cases")) == 2
    with pytest.raises(ResearchAdmissionError, match="admitted study identity"):
        runtime.observe_reviewed_research_sources(intake, **{**ids, "lineage_id": "other"}, **args)
    contents[root + "candidate.jsonl"] = b"corrupt\n"
    with pytest.raises(ResearchAdmissionError, match="bytes differ"):
        runtime.observe_reviewed_research_sources(intake, **ids, **args)
