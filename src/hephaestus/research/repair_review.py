"""Route validated research repairs to existing code-edit governance."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

from hephaestus.control.code_edit_workflow import CodeEditProposalWorkflow
from hephaestus.research.exact_span_repair import ExactSpanPatchProposer
from hephaestus.schemas.code_edit_proposal import CodeEditProposal
from hephaestus.schemas.research_repair import PatchModel, RepairBoundaryError, RepairScope
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def propose_reviewed_repair(
    *, artifacts: ArtifactStore, proposals: CodeEditProposalStore,
    scope: RepairScope, source_ref: str, verified_fault_ref: str,
    run_id: str, lineage_id: str, proposer: PatchModel,
) -> CodeEditProposal:
    """Create a review proposal for a trusted, hash-bound source and regression."""
    scope.validate()
    fault = read_verified_json(artifacts, verified_fault_ref)
    if (
        fault.get("schema") != "reviewed-code-failure.v1"
        or fault.get("status") != "verified_failed"
        or fault.get("run_id") != run_id
        or fault.get("lineage_id") != lineage_id
        or fault.get("base_commit") != scope.base_commit
        or fault.get("target_file") != scope.allowed_file
        or fault.get("fixed_test") != scope.fixed_test
        or fault.get("source_ref") != source_ref
    ):
        raise RepairBoundaryError("reviewed regression was not bound to the exact source and test")
    with artifacts.open(source_ref) as stream:
        original = stream.read(64_001)
    if len(original) > 64_000 or source_ref != "sha256:" + sha256(original).hexdigest():
        raise RepairBoundaryError("reviewed repair source bytes differ from their commitment")

    candidate = ExactSpanPatchProposer(proposer).propose(
        scope=scope, original=original, verified_fault=fault, evidence_ref=verified_fault_ref,
    )
    raw = canonical_bytes({
        "schema": "research-exact-span-candidate.v1",
        "base_commit": scope.base_commit,
        "candidate": asdict(candidate),
        "independent_verification": "pending",
    })
    reference = "sha256:" + sha256(raw).hexdigest()
    artifacts.put_bytes(raw, expected_hash=reference, media_type="application/json")
    proposal = CodeEditProposalWorkflow(proposals).create_proposal(
        run_id=run_id,
        lineage_id=lineage_id,
        requested_by="research-coder",
        purpose="Review a bounded repair for an independently observed code regression",
        target_files=[scope.allowed_file],
        proposed_diff_ref=reference,
        test_plan=[scope.fixed_test],
        evidence_refs=(verified_fault_ref, source_ref),
        rollback_plan="Revert the exact reviewed commit if approved and merged.",
        metadata={"requires_independent_red_green_verification": True},
    )
    if proposal.status == "blocked":
        raise RepairBoundaryError("code-edit governance rejected the bounded proposal")
    return proposal
