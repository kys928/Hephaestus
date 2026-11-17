"""Multiple anchored catalogs, ambiguity refusal, match provenance and no S3 listing."""
from __future__ import annotations
from hashlib import sha256

import pytest

from hephaestus.research.autonomous_study import ResearchAdmissionError
from hephaestus.research.reviewed_discovery import ReviewedEvidenceDiscovery
from hephaestus.storage.verified import canonical_bytes


def _catalog(tmp_path, key):
    path=tmp_path/(key+".json")
    sources=[]
    for role in ("original","candidate"):
        blob=(key+role).encode()
        sources.append({
            "source_id":key+"-"+role,
            "subject":key+"-scientific-run-"+role,
            "study_scope":key+"-evaluation-research-family",
            "source_uri":"runpod-s3://cviwpryzao/hephaestus/scientific/v1/"+key+"/"+role,
            "sha256_ref":"sha256:"+sha256(blob).hexdigest(),
            "expected_rows":1,"expected_bytes":len(blob),
            "role":role,"capability":"diagnosis_coverage_v1",
        })
    path.write_bytes(canonical_bytes({"schema":"research-source-catalog.v1","sources":sources}))
    return path,"sha256:"+sha256(path.read_bytes()).hexdigest()


def test_search_chooses_only_topic_matched_reviewed_source_family(tmp_path):
    index=ReviewedEvidenceDiscovery.open((_catalog(tmp_path,"tokenizer"),
                                          _catalog(tmp_path,"geometry")))
    result=index.search("Investigate Diagnostician reliability for tokenizer issues")
    assert result.original.study_scope.startswith("tokenizer-")
    assert "tokenizer" in result.matched_terms
    assert result.selection_kind=="deterministic_topic_match_not_scientific_gold"


def test_ambiguous_objective_must_not_guess_independent_ground_truth(tmp_path):
    index=ReviewedEvidenceDiscovery.open((_catalog(tmp_path,"tokenizer"),
                                          _catalog(tmp_path,"geometry")))
    with pytest.raises(ResearchAdmissionError,match="ambiguous"):
        index.search("Investigate Diagnostician reliability of its predictions")
    selected=index.search("Investigate Diagnostician reliability of its predictions",
                          required_scope="tokenizer-evaluation-research-family")
    assert selected.original.study_scope=="tokenizer-evaluation-research-family"


def test_unpinned_index_content_refused_before_retrieval(tmp_path):
    path,digest=_catalog(tmp_path,"tokenizer")
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ResearchAdmissionError,match="catalog"):
        ReviewedEvidenceDiscovery.open(((path,digest),))


def test_duplicate_reviewed_sources_are_not_counted_as_independent(tmp_path):
    value=_catalog(tmp_path,"tokenizer")
    with pytest.raises(ResearchAdmissionError,match="duplicate reviewed"):
        ReviewedEvidenceDiscovery.open((value,value))


def test_missing_matching_source_family_refused(tmp_path):
    index=ReviewedEvidenceDiscovery.open((_catalog(tmp_path,"tokenizer"),))
    with pytest.raises(ResearchAdmissionError,match="no source-verified"):
        index.search("Investigate Diagnostician reliability",required_scope="unknown")


def test_only_reviewed_pairs_and_content_hashes_are_returned(tmp_path):
    a=_catalog(tmp_path,"tokenizer")
    b=_catalog(tmp_path,"geometry")
    index=ReviewedEvidenceDiscovery.open((a,b))
    source=index.search("Investigate Diagnostician reliability for geometry failure")
    assert source.catalog.catalog_sha256==b[1]
    assert source.original.role=="original"
    assert source.candidate.role=="candidate"
    assert source.original.sha256_ref.startswith("sha256:")



def test_readonly_multi_catalog_cli_discovers_reviewed_pair_without_queue_or_s3(tmp_path,capsys):
    import json
    from hephaestus.research.cli import main
    first=_catalog(tmp_path,"geometry")
    second=_catalog(tmp_path,"tokenizer")
    root=tmp_path/"untouched-production-state"
    assert main([
        "--state-root",str(root),
        "discover","--objective","Investigate Diagnostician reliability of tokenizer coverage",
        "--catalog",str(first[0]),first[1],
        "--catalog",str(second[0]),second[1],
    ])==0
    output=json.loads(capsys.readouterr().out)
    assert output["study_scope"].startswith("tokenizer-")
    assert output["catalog_sha256"]==second[1]
    assert output["actual_source_bytes_verified"] is False
    assert output["independent_gold_label_established"] is False
    assert output["compute_authorized"] is False
    assert not root.exists()
