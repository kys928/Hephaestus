import json

import pytest

from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.schemas.research_repair import RepairBoundaryError, RepairScope
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def test_exact_span_research_proposal_goes_through_existing_review_store(tmp_path):
    runtime = ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path / "state", artifact_root=tmp_path / "evidence",
    )).build()
    scope = RepairScope(
        "a" * 40, "src/hephaestus/diagnosis/probes.py", "tests/test_targeted_post_failure_probes.py",
    )
    original = b"def classify():\n    return False\n\ndef other():\n    return None\n"
    source = runtime.artifact_store.put_bytes(original).artifact_ref
    fault = runtime.artifact_store.put_bytes(canonical_bytes({
        "schema": "reviewed-code-failure.v1", "status": "verified_failed",
        "run_id": "research-1", "lineage_id": "lineage-1",
        "base_commit": scope.base_commit, "target_file": scope.allowed_file,
        "fixed_test": scope.fixed_test, "source_ref": source,
        "regression_entrypoint": "classify",
    })).artifact_ref

    class FixturePatchText:
        before = "return False"

        def generate_patch_json(self, prompt):
            assert "def classify():" in prompt and "def other():" not in prompt
            return json.dumps({
                "old_text": self.before, "new_text": "return True",
                "hypothesis": "Correct the observed fixture regression.",
            })

    fixture = FixturePatchText()
    args = dict(scope=scope, source_ref=source, verified_fault_ref=fault,
                run_id="research-1", lineage_id="lineage-1", proposer=fixture)
    proposal = runtime.propose_research_repair(**args)
    assert proposal.status == "approval_required"
    persisted = CodeEditProposalStore(runtime.settings.state_root).get(proposal.proposal_id)
    assert persisted["target_files"] == [scope.allowed_file]
    candidate = read_verified_json(runtime.artifact_store, proposal.proposed_diff_ref)
    assert candidate["independent_verification"] == "pending"
    assert "return True" in candidate["candidate"]["replacement"]
    with runtime.artifact_store.open(source) as stream:
        assert stream.read() == original
    with pytest.raises(RepairBoundaryError, match="exact source and test"):
        runtime.propose_research_repair(**{**args, "lineage_id": "other"})
    fixture.before = "return None"
    with pytest.raises(RepairBoundaryError, match="ambiguous"):
        runtime.propose_research_repair(**args)
