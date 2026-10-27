"""Source admissibility never inferred from summaries, license text or newline flattening."""
import io
import importlib.util
import pathlib
import sys
import types
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def test_historical_campaign_status_never_replaces_original_state():
    historic=_script("recover_original_cpu_campaign_evidence")
    assert len(historic.MEMBERS)==2
    assert all(".sqlite" not in name for name in historic.MEMBERS)
    with pytest.raises(ValueError, match="ZIP byte identity"):
        historic.recover(b"not an original archive")


def test_hf_one_source_raw_newline_unknown_not_hidden():
    screen=_script("screen_single_hf_math_source_public_preview")
    class Encoded:
        def __init__(self, ids):
            self.ids=ids
    class FrozenTokenizer:
        def token_to_id(self, string):
            assert string=="<|unk|>"
            return 999
        def encode(self, text):
            # model the known original tokenization defect on newline
            return Encoded([999 if char=="\n" else 100 for char in text])
    meta={"sha":"a"*40,"cardData":{"license":"mit"}}
    splits={"splits":[{"dataset":screen.DATASET,"config":"default","split":"train"}]}
    first={"dataset":screen.DATASET,"config":"default","split":"train","rows":[
        {"row":{"problem_id":"p1","instruction":"Solve.","input":"1+2","output":"3"},"truncated_cells":[]},
        {"row":{"problem_id":"p1","instruction":"Solve quickly.","input":"1+2","output":"3"},"truncated_cells":[]},
    ]}
    r=screen.preview_review(meta,splits,first,FrozenTokenizer())
    assert r["valid_rows"]==2
    assert r["preview_counts"]["repeated_problem_id"]==1
    assert r["raw_newline_tokenizer"]["unk"]==4
    assert r["flattened_whitespace_control_only"]["unk"]==0
    assert not r["raw_newline_tokenizer"]["passes_unchanged_one_percent_ceiling"]
    assert r["recommendation"]=="HOLD_raw_tokenizer_above_one_percent"
    assert not r["training_authorized"] and not r["acquisition_approval"]


def test_source_bad_license_and_revision_fail_closed():
    screen=_script("screen_single_hf_math_source_public_preview")
    class Dummy:
        def token_to_id(self,n):return 999
    meta={"sha":"a"*40,"cardData":{"license":"unknown"}}
    with pytest.raises(ValueError,match="MIT"):
        screen.preview_review(meta,{"splits":[]},{},Dummy())


def test_s3_locator_honestly_reports_page_limit_and_possible_sqlite():
    search=_script("locate_original_campaign_s3_state")
    class Fake:
        def __init__(self):self.n=0
        def list_objects_v2(self,**kwargs):
            self.n+=1
            if self.n==1:
                return {"Contents":[{"Key":"old/state/production.sqlite3","Size":203}],
                        "IsTruncated":True,"NextContinuationToken":"t"}
            return {"Contents":[{"Key":"public/text.txt","Size":100}],
                    "IsTruncated":False}
    report=search.inspect(Fake(),"bucket")
    assert report["entire_bucket_listed"]
    assert report["candidate_count"]==1
    assert report["candidate_state_objects"][0]["key"].endswith(".sqlite3")
    assert report["candidate_authenticity_verified"] is False
    assert report["historical_receipts_recovered"] is False
