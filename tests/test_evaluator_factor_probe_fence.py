"""No-cost test: strict exact-only Markdown JSON fence interpretation."""
import json
import pytest
from assess_evaluator_factor_probe_fenced import normalize_exact_fenced_object,normalize_fenced_singleton_array
from evaluator_semantic_factor_probe_protocol import parse_factor_json

FENCE=chr(96)*3

@pytest.fixture
def payload():
    return {"gate_pass":True,"complete":True,"provenance_valid":True,
        "effect":"neutral","variance_stable":True,"recheck_satisfied":False}

def test_only_complete_json_fences_are_normalizable(payload):
    raw=json.dumps(payload)
    assert parse_factor_json(raw) is not None
    assert normalize_exact_fenced_object(FENCE+"json\n"+raw+"\n"+FENCE)==raw
    assert normalize_exact_fenced_object(FENCE+"\n"+raw+"\n"+FENCE)==raw

@pytest.mark.parametrize("mode",["extra_prefix","extra_suffix","wrong_keys","not_fenced","truncated"])
def test_partial_arbitrary_or_wrong_json_rejected(payload,mode):
    raw=json.dumps(payload)
    if mode=="extra_prefix":text="Prose: "+FENCE+"json\n"+raw+"\n"+FENCE
    elif mode=="extra_suffix":text=FENCE+"json\n"+raw+"\n"+FENCE+" extra prose"
    elif mode=="wrong_keys":text=FENCE+"json\n"+json.dumps({**payload,"unexpected":1})+"\n"+FENCE
    elif mode=="not_fenced":text=raw
    else:text=FENCE+"json\n"+raw[:15]
    assert normalize_exact_fenced_object(text) is None


def test_only_one_fully_fenced_singleton_array_is_accepted(payload):
    content=json.dumps([payload],indent=2)
    fenced=FENCE+"json\n"+content+"\n"+FENCE
    normalized=normalize_fenced_singleton_array(fenced)
    assert normalized is not None
    assert parse_factor_json(normalized) is not None
    assert json.loads(normalized)==payload
    assert normalize_exact_fenced_object(fenced) is None


def test_singleton_array_rejects_invalid_wrappers_and_extra_items(payload):
    def wrap(content):
        return FENCE+"json\n"+content+"\n"+FENCE
    assert normalize_fenced_singleton_array(wrap(json.dumps([payload,payload]))) is None
    assert normalize_fenced_singleton_array("Hello "+wrap(json.dumps([payload]))) is None
    assert normalize_fenced_singleton_array(wrap(json.dumps([{"effect":"neutral"}]))) is None
    assert normalize_fenced_singleton_array(wrap(json.dumps(payload))) is None
    assert normalize_fenced_singleton_array(wrap(json.dumps([payload]))+" Extra") is None
