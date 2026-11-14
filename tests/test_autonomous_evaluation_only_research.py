"""Real read-only bytes and persistent artifact receipts; fake inference is explicit."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import pytest

from hephaestus.research.autonomous_study import (
    EvaluationOnlyResearchJob, ReadOnlyS3SourceReader, ResearchAdmissionError,
    ResearchObjective, SourceCatalog, VerifiedSourceAcquirer,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes


def _pack():
    return {
        "generation_probes": [{"prompt": "Reply: alpha beta gamma"}],
        "structure_tests": [{"prompt": "Return JSON with answer and confidence"}],
        "length_termination_checks": [{"prompt": "Limit output to twelve words"}],
    }


def _original():
    return (b'{"text":"plain prose original"}\n' * 3)


def _candidate():
    rows = [
        {"text":"<|prompt|>\nWhat is Mars?\n<|target|>\nA planet"},
        {"text":"<|prompt|>\nReturn JSON\n<|target|>\nnot a JSON object"},
        {"text":"<|prompt|>\nBroken\n<|target|>\n"},
        {"text":"unstructured extra"},
    ]
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


@dataclass
class MemoryReadOnlyReader:
    objects: dict[str, bytes]
    requests: list[str]

    def read(self, uri: str, *, maximum_bytes: int) -> bytes:
        self.requests.append(uri)
        data = self.objects[uri]
        if len(data) > maximum_bytes:
            raise ResearchAdmissionError("oversize")
        return data


@dataclass
class FakeOnlyForCpuUnitTests:
    artifacts: FileSystemArtifactStore
    calls: list[tuple[str, int]]

    def observe(self, *, objective, observation, seed, run_id, lineage_id):
        self.calls.append((observation["source_id"], seed))
        receipt = self.artifacts.put_bytes(canonical_bytes({
            "schema": "declared-fake-inference-receipt-for-tests",
            "source": observation["observation_ref"], "seed": seed,
        })).artifact_ref
        return {"decision": "inconclusive", "action": "collect_more_evidence",
                "primary_variable": "none", "confidence": 0.2,
                "cited_refs": [observation["observation_ref"]],
                "receipt_ref": receipt, "model_id": "fake-not-certified"}


def _case_catalog(tmp_path, originals=None):
    sources = [_original(), _candidate()] if originals is None else originals
    catalog = {"schema": "research-source-catalog.v1", "sources": []}
    mapping = {}
    for role, data in zip(("original", "candidate"), sources):
        uri = f"runpod-s3://test-bucket/hephaestus/scientific/v1/{role}/dataset.jsonl"
        mapping[uri] = data
        catalog["sources"].append({
            "source_id": role, "subject": role, "study_scope": "real-matched-source-family",
            "source_uri": uri, "sha256_ref": "sha256:"+hashlib.sha256(data).hexdigest(),
            "expected_bytes": len(data),
            "expected_rows": len(data.splitlines()), "role": role,
            "capability": "diagnosis_coverage_v1",
        })
    path=tmp_path/"reviewed-catalog.json"
    path.write_bytes(canonical_bytes(catalog))
    digest="sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()
    return path,digest,mapping


def _job(tmp_path, originals=None):
    path, digest, contents = _case_catalog(tmp_path, originals)
    artifacts = FileSystemArtifactStore(tmp_path/"immutable")
    repository = SQLiteStateRepository(tmp_path/"state.db")
    reader = MemoryReadOnlyReader(contents, [])
    observer = FakeOnlyForCpuUnitTests(artifacts, [])
    job = EvaluationOnlyResearchJob(
        catalog=SourceCatalog.load(path, independently_pinned_sha256=digest),
        acquirer=VerifiedSourceAcquirer(reader, artifacts, repository),
        observer=observer, repository=repository, artifacts=artifacts,
        eval_pack=_pack(),
    )
    return job, reader, observer, path


def test_natural_objective_triggers_verified_evaluation_only_job_without_case_list(tmp_path):
    job,reader,observer,_=_job(tmp_path)
    result=job.run("Investigate Diagnostician reliability and evidence coverage",
                   run_id="investigation-1", lineage_id="research-lineage")
    assert result["schema"]=="research-evaluation-only-report.v1"
    assert result["role_invocations"]==6
    assert result["verified_case_count"]==2
    assert len(reader.requests)==2
    assert len(observer.calls)==6
    assert result["training_performed"] is False
    assert result["model_modified"] is False
    assert result["independent_scientific_certification"] is False
    assert result["observed_source_difference"]["instruction_pairs_delta"]==2
    assert result["observed_source_difference"]["structured_targets_delta"]==0
    assert result["observed_source_difference"]["original_gap"] is True
    assert result["observed_source_difference"]["candidate_gap"] is True
    assert result["decision_distributions"]=={"original":{"inconclusive":3},"candidate":{"inconclusive":3}}
    repeated=job.run("Investigate Diagnostician reliability and evidence coverage",
                     run_id="investigation-1",lineage_id="research-lineage")
    assert repeated==result
    assert len(reader.requests)==2
    assert len(observer.calls)==6


@pytest.mark.parametrize("bad_question", [
    "", "x", "Build me a billion parameter model",
    "Mutate the frozen eval pack and claim victory", "Please delete all checkpoints",
])
def test_unsupported_or_unbounded_objective_does_not_launch_anything(tmp_path,bad_question):
    job,reader,observer,_=_job(tmp_path)
    with pytest.raises(ResearchAdmissionError):
        job.run(bad_question,run_id="reject",lineage_id="research-lineage")
    assert reader.requests==[] and observer.calls==[]


def test_independently_pinned_catalog_bytes_required(tmp_path):
    path,digest,_=_case_catalog(tmp_path)
    assert len(SourceCatalog.load(path,independently_pinned_sha256=digest).sources)==2
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ResearchAdmissionError,match="catalog"):
        SourceCatalog.load(path,independently_pinned_sha256=digest)


def test_source_tamper_fails_before_inference_or_persistence(tmp_path):
    job,reader,observer,_=_job(tmp_path)
    key=next(uri for uri in reader.objects if "/candidate/" in uri)
    reader.objects[key]+=b'{"text":"forged"}\n'
    with pytest.raises(ResearchAdmissionError,match="source bytes differ"):
        job.run("Investigate Diagnostician reliability and evidence coverage",
                run_id="bad-source",lineage_id="research-lineage")
    assert observer.calls==[]
    assert job.repository.all("research_evaluation_studies")==[]


def test_wrong_source_family_fails_before_read(tmp_path):
    path,digest,contents=_case_catalog(tmp_path)
    doc=json.loads(path.read_bytes())
    doc["sources"][1]["study_scope"]="unrelated-granite-eight-state-ontology"
    path.write_bytes(canonical_bytes(doc))
    digest="sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ResearchAdmissionError,match="uniquely matched"):
        SourceCatalog.load(path,independently_pinned_sha256=digest).select(
            ResearchObjective.from_text("Investigate Diagnostician reliability and evidence coverage"))


def test_source_observation_receipt_must_be_present_and_exactly_cited(tmp_path):
    job,reader,observer,_=_job(tmp_path)
    class BadObserver:
        def observe(self, *, objective, observation, seed, run_id, lineage_id):
            return {"decision":"inconclusive", "receipt_ref":"sha256:"+"0"*64,
                    "cited_refs":[observation["observation_ref"]]}
    job.observer=BadObserver()
    with pytest.raises(ResearchAdmissionError,match="receipt"):
        job.run("Investigate Diagnostician reliability and evidence coverage",
                run_id="bad-observer",lineage_id="research-lineage")


def test_s3_reader_exact_bucket_and_prefix_no_listing_or_write():
    calls=[]
    class FakeS3:
        def get_object(self,*,Bucket,Key):
            calls.append((Bucket,Key))
            return {"Body":io.BytesIO(b"safe")}
    reader=ReadOnlyS3SourceReader(FakeS3(),"reviewed")
    assert reader.read("runpod-s3://reviewed/hephaestus/scientific/v1/original",
                       maximum_bytes=4)==b"safe"
    assert calls==[("reviewed","hephaestus/scientific/v1/original")]
    for bad in (
        "runpod-s3://untrusted/hephaestus/scientific/v1/data",
        "runpod-s3://reviewed/other-path",
        "runpod-s3://reviewed/hephaestus/scientific/v1/../../private",
        "https://example.com/model",
    ):
        with pytest.raises(ResearchAdmissionError):
            reader.read(bad,maximum_bytes=20)
    assert len(calls)==1


def test_model_must_cite_exact_authenticated_source_and_no_fabricated_gold(tmp_path):
    job,_,_,_=_job(tmp_path)
    class MisattributedObserver:
        def observe(self, *, objective, observation, seed, run_id, lineage_id):
            ref=job.artifacts.put_bytes(b"{}").artifact_ref
            return {"decision":"evaluation_integrity", "receipt_ref":ref,
                    "cited_refs":["unrelated-source"]}
    job.observer=MisattributedObserver()
    with pytest.raises(ResearchAdmissionError,match="receipt"):
        job.run("Investigate Diagnostician reliability and evidence coverage",
                run_id="bad-citation",lineage_id="research-lineage")
