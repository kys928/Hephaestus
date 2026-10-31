"""Verify actual pinned CPU causal LM proposals from guarded scientific evidence."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.llm.role_contract import RoleTask, RoleEvidence
from hephaestus.llm.cpu_causal_research import (
    CpuCausalProposalScorer, CANDIDATES, MODEL_ID, MODEL_REVISION,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def _source(artifacts):
    source = artifacts.put_bytes(
        canonical_bytes({
            "schema":"cpu-causal-research-source.v1",
            "run_id":"cpu-causal-1", "lineage_id":"lineage-1",
            "observed_evaluation_delta":0,
            "scientific_certification":"not_certified",
            "provenance":"synthetic_test_only",
        }), media_type="application/json")
    ref = source.artifact_ref
    registry = EvidenceRegistry({ref:EvidenceRecord(
        ref=ref, kind="synthetic_test_only",
        run_id="cpu-causal-1", lineage_id="lineage-1",
        sha256=ref.removeprefix("sha256:"),
        observed_sha256=ref.removeprefix("sha256:"),
    )})
    return ref,registry


@pytest.mark.real_cpu
def test_actual_pinned_pretrained_causal_lm_proposes_five_role_reviews(tmp_path):
    try:
        import torch
        import transformers
    except ImportError:
        pytest.skip("actual PyTorch and Transformers required")
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    ref,registry = _source(artifacts)
    scorer = CpuCausalProposalScorer.load(
        artifacts, cache_dir=tmp_path / "hf-cache")
    result = []
    for role in CANDIDATES:
        task = RoleTask(
            role=role,
            situation=(
                "The synthetic smoke evaluation has zero proven improvement, "
                "no approved live promotion and no independent contamination audit. "
                "What evidence or governed review is needed next?"
            ),
            evidence=(RoleEvidence(ref,
                     "The source is synthetic, with no external data custody and no certified effect."),),
            run_id="cpu-causal-1", lineage_id="lineage-1", stage_name="smoke_test",
        )
        context = DeterministicRoleContext(
            "cpu-causal-1", "lineage-1", "smoke_test", approval_status="none")
        outcome = scorer.propose(task, registry, context)
        assert outcome["identity"]["model_id"] == MODEL_ID
        assert outcome["identity"]["revision"] == MODEL_REVISION
        assert outcome["identity"]["model_backend"] != "Fake"
        assert outcome["identity"]["model_weights_file_sha256"].startswith("sha256:")
        assert outcome["selected_candidate"] in outcome["candidates"]
        assert len(outcome["candidates"]) == 3
        assert all(math.isfinite(x["mean_conditional_log_likelihood"])
                   for x in outcome["candidates"])
        assert not outcome["action_executed"]
        assert not outcome["production_role_certified"]
        assert not outcome["scientific_quality_certified"]
        doc = read_verified_json(artifacts, outcome["inference_receipt_ref"])
        assert doc["role"] == role
        result.append({
            "role": role, "selected": outcome["selected_candidate"],
            "receipt_ref": outcome["inference_receipt_ref"],
            "candidate_scores":outcome["candidates"],
        })
    assert len(result)==5
    report = os.environ.get("HEPHAESTUS_REAL_CPU_LM_REPORT")
    if report:
        path = Path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema":"cpu-pretrained-causal-lm-role-execution.v1",
            "model_id":MODEL_ID, "revision":MODEL_REVISION,
            "weights_file_sha256":outcome["identity"]["model_weights_file_sha256"],
            "cpu_threads":torch.get_num_threads(),
            "torch_version":torch.__version__,
            "transformers_version":transformers.__version__,
            "source_type":"synthetic_smoke_evidence",
            "actual_pretrained_causal_lm_inference":True,
            "frozen_five_role_production_model":False,
            "full_45_operation_campaign":False,
            "role_proposals":result,
            "independent_scientific_certification":False,
            "no_promotions":True,
        },indent=2,sort_keys=True)+"\n")


def test_causal_lm_approvals_can_never_be_inferred_from_candidate_vocabulary():
    assert "approved" not in {x[0] for x in CANDIDATES["judge"]}
    assert "promote_checkpoint" not in {x[1] for x in CANDIDATES["judge"]}


def test_causal_lm_source_verification_fails_closed_without_loading_model(tmp_path):
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    ref,registry=_source(artifacts)
    scorer=CpuCausalProposalScorer(None,None,artifacts,tmp_path)
    task=RoleTask("planner","What next?",(RoleEvidence(ref,"source"),),
                  "a-different-run","lineage-1","smoke_test")
    context=DeterministicRoleContext("a-different-run","lineage-1","smoke_test")
    with pytest.raises(PermissionError, match="verified source"):
        scorer.propose(task,registry,context)
