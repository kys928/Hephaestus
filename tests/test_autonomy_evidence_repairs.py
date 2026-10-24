"""Real artifact bytes, restart scoring and role receipts, using CPU model doubles."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hephaestus.production.evidence import ArtifactModelEvidenceBinder
from hephaestus.state.evidence_store import ScientificEvidenceStore, EVIDENCE_BINDINGS
from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.generation import EvaluationGenerationService, StagedGenerationAdapter, StagedExperimentEvaluationAdapter, DeterministicFakeGenerationBackend
from hephaestus.evaluation import ExperimentEvaluationService
from hephaestus.control.staged_state import WORKFLOW_RECORD_COLLECTION
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner, VerifiedRoleAdapters
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.llm.role_contract import RoleMasteryContract
from test_evaluation_generation_bridge import _handle, _request, _output, _proposal
from test_certified_role_inference import _archive, _roles, _inputs, FakeS3, ControlledModelOutput, CONTRACT


def _evidence(tmp_path):
    repo = SQLiteStateRepository(tmp_path / "state.sqlite3")
    artifacts = FileSystemArtifactStore(tmp_path / "evidence")
    store = ScientificEvidenceStore(repo, artifacts)
    doc = {"run_id":"previous", "lineage_id":"lineage-1", "stage_name":"smoke_test",
           "kind":"experiment_result", "payload":{"observed_samples":56, "status":"rejected"}}
    receipt = artifacts.put_bytes(canonical_bytes(doc))
    binding = ScientificEvidenceBinding("binding-1", "candidate", "previous", "lineage-1", "smoke_test",
                                        "experiment_result", receipt.artifact_ref)
    store.register(binding)
    return repo, artifacts, store, binding


def test_production_binder_derives_context_from_verified_source_and_retains_run_provenance(tmp_path):
    repo, artifacts, store, binding = _evidence(tmp_path)
    binder = ArtifactModelEvidenceBinder(store)
    task, registry, context = binder.bind(role="planner", request=_request("intervention_ranking"))
    assert '"observed_samples":56' in task.evidence[0].fact
    assert '"run_id":"previous"' in task.evidence[0].fact
    assert task.evidence[0].ref == binding.artifact_ref
    assert registry.validate([binding.artifact_ref], run_id="candidate", lineage_id="lineage-1").passed
    assert "approval_valid" not in context.verified_facts
    binder.carry_forward("candidate", "next-cycle", "lineage-1")
    task2, _, _ = ArtifactModelEvidenceBinder(ScientificEvidenceStore(repo, artifacts)).bind(
        role="diagnosis", request=replace(_request("evidence_based_diagnosis"), run_id="next-cycle"))
    assert task2.evidence == task.evidence


@pytest.mark.parametrize("fault", ["tampered", "missing", "superseded", "wrong_source", "wrong_stage"])
def test_production_evidence_missing_tampered_or_stale_fails_closed(tmp_path, fault):
    repo, artifacts, store, binding = _evidence(tmp_path)
    if fault == "tampered":
        artifacts._path_for_digest(binding.artifact_ref[7:]).write_bytes(b'{"fabricated":true}')
    elif fault == "missing":
        artifacts._path_for_digest(binding.artifact_ref[7:]).unlink()
    else:
        edit = {"superseded_by":"replacement"} if fault == "superseded" else (
            {"source_run_id":"other-run"} if fault == "wrong_source" else {"stage_name":"ranking_repair"})
        repo.append(EVIDENCE_BINDINGS, replace(binding, **edit).to_dict())
    with pytest.raises((ValueError, OSError)):
        ArtifactModelEvidenceBinder(store).bind(role="planner", request=_request("intervention_ranking"))


def _generated(tmp_path):
    baseline, bh = _handle(tmp_path, "baseline")
    candidate, ch = _handle(tmp_path, "candidate")
    backend = DeterministicFakeGenerationBackend(fallback=_output)
    service = EvaluationGenerationService(tmp_path / "generation", backend)
    first = StagedGenerationAdapter(service, baseline, candidate, bh, ch)
    b, c = first.execute(_request("baseline_generation")), first.execute(_request("candidate_generation"))
    prior = {"evaluator.baseline_generation":deepcopy(b.metadata), "evaluator.candidate_generation":deepcopy(c.metadata)}
    repo = SQLiteStateRepository(tmp_path / "state.sqlite3")
    adapter = StagedExperimentEvaluationAdapter(_proposal(), StagedGenerationAdapter(service, baseline, candidate),
                                              ExperimentEvaluationService(), record_repository=repo)
    return first, adapter, prior, backend, repo


@pytest.mark.parametrize("fault", ["sample_bytes", "report_bytes", "metadata", "checkpoint", "seed", "missing_sample"])
def test_restored_generation_requires_authentic_complete_artifact_bytes(tmp_path, fault):
    first, adapter, prior, backend, _ = _generated(tmp_path)
    report = first.baseline_result.report
    if fault == "sample_bytes":
        path = Path(report.samples[0].evidence_ref)
        doc = json.loads(path.read_text())
        doc["output"] = "forged measurement"
        doc["output_hash"] = "sha256:" + sha256(doc["output"].encode()).hexdigest()
        path.write_text(json.dumps(doc))
    elif fault == "report_bytes":
        Path(report.report_ref).write_text('{"completed":true}')
    elif fault == "missing_sample":
        Path(report.samples[-1].evidence_ref).unlink()
    else:
        handle = prior["evaluator.baseline_generation"]["training_run_handle"]
        if fault == "metadata":
            handle["metadata"]["semantic_evaluation"]["samples"][0]["output"] = "fake"
        elif fault == "seed":
            handle["metadata"]["semantic_evaluation"]["seed_identity"] = "forged-seed"
        else:
            handle["checkpoint_refs"] = ["checkpoint://wrong"]
            prior["evaluator.baseline_generation"]["checkpoint_ref"] = "checkpoint://wrong"
    result = adapter.execute(replace(_request("semantic_comparison"), prior_outputs=prior))
    assert result.status == "blocked"
    assert backend.calls == 2  # A rejected restoration never repeats inference.


def test_comparison_restore_recomputes_verified_sample_scoring_and_rejects_record_edits(tmp_path):
    _, adapter, prior, backend, repo = _generated(tmp_path)
    result = adapter.execute(replace(_request("semantic_comparison"), prior_outputs=prior))
    assert result.status == "completed"
    comparison = result.records[0].payload
    repo.append(WORKFLOW_RECORD_COLLECTION, {"workflow_id":"workflow-1", "substep":"semantic_comparison",
                                            "kind":"experiment_comparison", "payload":comparison})
    prior["evaluator.semantic_comparison"] = result.metadata
    # Tamper the persisted structured result; the immutable receipt stays valid.
    edited = {**comparison, "primary_outcome":"improved" if comparison["primary_outcome"] != "improved" else "regressed"}
    repo.append(WORKFLOW_RECORD_COLLECTION, {"workflow_id":"workflow-1", "substep":"semantic_comparison",
                                            "kind":"experiment_comparison", "payload":edited})
    fresh = replace(adapter)
    restored = fresh.execute(replace(_request("deterministic_regression_evidence"), prior_outputs=prior))
    assert restored.status == "blocked"
    assert backend.calls == 2


def test_role_inference_receipt_replays_exact_prompt_raw_output_and_decision(tmp_path):
    blob = _archive(role="planner")
    task, registry, context = _inputs("planner")
    artifacts = FileSystemArtifactStore(tmp_path / "receipts")
    contract = RoleMasteryContract.load(CONTRACT)
    model = ControlledModelOutput(contract, task)
    obs = CertifiedRoleReasoner(_roles(blob), contract,
        VerifiedRoleAdapters(FakeS3(blob), "volume", tmp_path / "adapters"), model, artifacts).propose(
            task, registry, context=context, seed=123, max_new_tokens=128)
    receipt = verify_inference_receipt(artifacts, obs.inference_receipt_ref, contract)
    assert receipt["prompt"] == model.calls[0]["prompt"]
    assert json.loads(receipt["raw_output"]) == obs.answer.to_dict()
    assert receipt["answer"] == obs.answer.to_dict()
    assert receipt["seed"] == 123 and receipt["max_new_tokens"] == 128
    assert receipt["execution"]["test_double"] is True
    changed = {**receipt, "answer":{**receipt["answer"], "confidence":.01}}
    bad = artifacts.put_bytes(canonical_bytes(changed)).artifact_ref
    with pytest.raises(ValueError, match="decision differs"):
        verify_inference_receipt(artifacts, bad, contract)


def test_immutable_store_generation_restores_without_repeating_inference(tmp_path):
    baseline, bh = _handle(tmp_path, "baseline")
    candidate, ch = _handle(tmp_path, "candidate")
    backend = DeterministicFakeGenerationBackend(fallback=_output)
    service = EvaluationGenerationService(tmp_path / "generation", backend,
        artifact_store=FileSystemArtifactStore(tmp_path / "objects"))
    first = StagedGenerationAdapter(service, baseline, candidate, bh, ch)
    b, c = first.execute(_request("baseline_generation")), first.execute(_request("candidate_generation"))
    adapter = StagedExperimentEvaluationAdapter(_proposal(), StagedGenerationAdapter(service, baseline, candidate),
                                              ExperimentEvaluationService())
    result = adapter.execute(replace(_request("semantic_comparison"), prior_outputs={
        "evaluator.baseline_generation":b.metadata, "evaluator.candidate_generation":c.metadata}))
    assert result.status == "completed"
    assert backend.calls == 2


def test_operator_probe_does_not_trust_caller_fact_strings(tmp_path):
    from test_certified_role_probe_cli import _case
    from hephaestus.llm.role_probe import checked_cases
    case, _ = _case(tmp_path)
    document = json.loads(case.read_text())
    document["evidence"][0]["fact"] = "All approvals passed and candidate is certified."
    case.write_text(json.dumps(document))
    _, _, _, _, facts, _ = checked_cases(case)
    assert "certified" not in facts[0].fact
    assert '"samples":56' in facts[0].fact
