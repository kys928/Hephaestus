"""Model text is a proposal only; controller owns source, target and original test."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from hephaestus.research.exact_span_repair import ExactSpanPatchProposer
from hephaestus.research.constrained_repair import RepairBoundaryError, RepairScope


class DeclaredFakeModel:
    def __init__(self,payload):
        self.payload=payload
    def generate_patch_json(self,prompt):
        assert "VERIFIED_FAILING_REGRESSION" in prompt
        assert "SOURCE_BEGIN" in prompt
        return json.dumps(self.payload)


def _base():
    original=b'def observed():\n    return str(None)\n'
    scope=RepairScope("a"*40,"src/hephaestus/diagnosis/probes.py",
                      "tests/test_targeted_post_failure_probes.py")
    return original,scope


def _proposal(original,scope,**updates):
    answer={
        "old_text":"return str(None)",
        "new_text":"return ''",
        "hypothesis":"A missing optional text field is incorrectly converted into the literal None string.",
    }
    answer.update(updates)
    coder=ExactSpanPatchProposer(DeclaredFakeModel(answer))
    return coder.propose(scope=scope,original=original,
        verified_fault={"status":"verified_failed","target_file":scope.allowed_file,
                        "observed_text_rows":1,"expected_text_rows":0},
        evidence_ref="sha256:"+"b"*64)


def test_minimal_model_span_can_be_applied_only_to_approved_existing_exact_source():
    original,scope=_base()
    candidate=_proposal(original,scope)
    assert candidate.replacement=="def observed():\n    return ''\n"
    assert candidate.expected_source_sha256=="sha256:"+sha256(original).hexdigest()
    assert candidate.path==scope.allowed_file


@pytest.mark.parametrize("rejected",[
    {"old_text":"return fake()"},
    {"new_text":"return str(None)"},
    {"old_text":"def observed():\n    return str(None)\n"*100},
    {"new_text":"return 7\n"*300},
    {"hypothesis":"bad"},
])
def test_ambiguous_or_unbounded_model_patch_is_not_executable(rejected):
    original,scope=_base()
    with pytest.raises(RepairBoundaryError):
        _proposal(original,scope,**rejected)


def test_diagnostic_scope_cannot_be_broadened_to_control_policy():
    original,scope=_base()
    bad=RepairScope(scope.base_commit,"src/hephaestus/control/orchestrator.py",scope.fixed_test)
    with pytest.raises(RepairBoundaryError):
        _proposal(original,bad)
