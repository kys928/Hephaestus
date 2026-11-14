"""Precommitted format-only response citation guidance, not a scientific answer."""
import json
import pytest

from hephaestus.development.prometheus_scientific_decision import (
    REF_PARENT, REF_D, _exact_source_citation_guidance,
)
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError
from hephaestus.llm.role_contract import RoleEvidence


def frozen_sources():
    return (RoleEvidence(REF_PARENT, "Previously exposed original six-state observation"),
            RoleEvidence(REF_D, "Different previously exposed eight-state observation"))


def test_citation_copy_guidance_contains_only_real_original_hash_identifiers():
    phrase = _exact_source_citation_guidance(frozen_sources())
    assert REF_PARENT in phrase and REF_D in phrase
    assert phrase.count(REF_PARENT) == 1 and phrase.count(REF_D) == 1
    assert "never a description" in phrase
    assert "non-interchangeable" in phrase
    assert "collect_more_evidence" not in phrase
    assert "repair_evaluation" not in phrase
    assert "resume_training" not in phrase


def test_citation_copy_guidance_refuses_missing_or_forged_identity():
    with pytest.raises(PinnedEvidenceError):
        _exact_source_citation_guidance((frozen_sources()[0],))
    with pytest.raises(PinnedEvidenceError):
        _exact_source_citation_guidance(
            (*frozen_sources(), RoleEvidence("sha256:"+"a"*64, "injected")))
    with pytest.raises(PinnedEvidenceError):
        _exact_source_citation_guidance(
            (RoleEvidence("proof:alias","unverified source"),frozen_sources()[1]))


def test_frozen_role_contract_still_rejects_natural_language_reference():
    from hephaestus.llm.role_contract import RoleMasteryContract
    c=RoleMasteryContract.load("configs/models/hephaestus_role_mastery_runtime_v1.json")
    response={
      "decision": "inconclusive",
      "action": "collect_more_evidence",
      "primary_variable": "evidence",
      "confidence": .5,
      "evidence_refs": ["An original development report instead of its hash"],
      "uncertainties": [],
      "rationale": "The observations do not establish novel parent generalization.",
    }
    with pytest.raises(ValueError, match="evidence"):
        c.parse("diagnosis",json.dumps(response),{REF_PARENT,REF_D})
    response["evidence_refs"]=[REF_PARENT]
    assert c.parse("diagnosis",json.dumps(response),{REF_PARENT,REF_D}).evidence_refs==(REF_PARENT,)
