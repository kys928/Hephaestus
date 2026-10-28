#!/usr/bin/env python3
"""Execute or resume one genuine bounded internal CPU native 45-operation campaign."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
from hashlib import sha256

from run_internal_data_intelligence import prepare_inputs
from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import WORKFLOW_STATE_COLLECTION, StagedWorkflowState
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.data.intelligence import AutonomousDatasetIntelligence
from hephaestus.data.internal_screening import InternalSourceScanner
from hephaestus.data.generated_validation import OriginalTaskSemanticChecker
from hephaestus.data.generated_factory import OriginalTaskRegistryProvider
from hephaestus.providers.datasets.approved_registry import ApprovedHubRegistryProvider, EvidenceBoundSourceInspector
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest
from hephaestus.schemas.data_intelligence import DiscoveryBudget
from hephaestus.schemas.cpu_research import CpuResearchExecutionProfile
from hephaestus.schemas.cpu_campaign_config import CpuCampaignConfig
from hephaestus.policy.internal_research import InternalDatasetAdmissionPolicy, put_record
from hephaestus.state.prospective_evaluation import ProspectiveEvaluationStore
from hephaestus.state.cpu_research import CpuResearchState
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json, canonical_bytes
from hephaestus.production.durable_campaign import create_fresh_campaign_workspace, verify_campaign_workspace, CampaignWorkspace
from hephaestus.production.cpu_research_context import CpuResearchContext
from hephaestus.production.cpu_research import compose_cpu_research_dependencies, verify_cpu_research_trace
from hephaestus.llm.cpu_causal_research import CpuCausalProposalScorer, MODEL_ID, MODEL_REVISION


def load_context(root, *, cache_dir, resume=False, campaign_config_path=None):
    if not resume:
        if campaign_config_path is None:
            raise ValueError("new CPU research campaign requires an explicit campaign configuration")
        spec = CpuCampaignConfig.from_file(campaign_config_path)
        workspace = create_fresh_campaign_workspace(root, **spec.identity)
        artifacts = FileSystemArtifactStore(workspace.artifacts_path)
        scorer = CpuCausalProposalScorer.load(artifacts, cache_dir=cache_dir)
        admission, candidate, review, material, scope, report = prepare_inputs(root,
            tokenizer_path=scorer.snapshot_dir / "tokenizer.json",
            include_public=spec.include_public, campaign_config=spec)
        # prepare_inputs uses exactly the native campaign's state.sqlite/artifacts directory.
        profile = CpuResearchExecutionProfile("bounded-original-cpu-research-v1", admission.policy_ref,
            MODEL_ID, MODEL_REVISION, {p.name: "sha256:" + sha256(p.read_bytes()).hexdigest()
                for p in sorted(scorer.snapshot_dir.iterdir()) if p.is_file()},
            admission.policy().tokenizer_sha256, spec.maximum_training_steps,
            spec.maximum_cpu_seconds, seed=spec.cpu_seed)
        profile.validate()
        profile_ref = put_record(artifacts, profile.to_dict())
        discovery = read_verified_json(artifacts, report["decision"]["evidence_refs"][0])
        spec_ref = put_record(artifacts, spec.to_dict())
        manifest = {"schema": "cpu-research-campaign-context.v1", "profile_ref": profile_ref,
            "campaign_spec_ref": spec_ref,
            "bootstrap_ref": workspace.bootstrap_artifact_ref, "candidate": candidate.to_dict(),
            "review_ref": review, "materialization_ref": material, "scope": scope,
            "data_report_ref": put_record(artifacts, report), "search_request": discovery["request"],
            "include_public": spec.include_public, "started_at": datetime.now(timezone.utc).isoformat()}
        context_ref = put_record(artifacts, manifest)
        (root / "context_ref.txt").write_text(context_ref + "\n")
    artifacts = FileSystemArtifactStore(root / "artifacts")
    manifest = read_verified_json(artifacts, (root / "context_ref.txt").read_text().strip())
    profile = CpuResearchExecutionProfile(**read_verified_json(artifacts, manifest["profile_ref"]))
    profile.validate()
    spec = CpuCampaignConfig.from_dict(read_verified_json(artifacts, manifest["campaign_spec_ref"]))
    if campaign_config_path is not None and CpuCampaignConfig.from_file(campaign_config_path) != spec:
        raise PermissionError("CPU campaign config differs from the immutable original specification")
    if (profile.maximum_training_steps != spec.maximum_training_steps or
            profile.maximum_cpu_seconds != spec.maximum_cpu_seconds or profile.seed != spec.cpu_seed):
        raise PermissionError("CPU profile differs from the immutable campaign limits or seed")
    workspace = CampaignWorkspace(root, **spec.identity, bootstrap_artifact_ref=manifest["bootstrap_ref"])
    verify_campaign_workspace(workspace)
    if resume:
        scorer = CpuCausalProposalScorer.load(artifacts, cache_dir=cache_dir, local_files_only=True)
    repository = SQLiteStateRepository(workspace.state_path)
    evaluation = ProspectiveEvaluationStore(repository, artifacts)
    scanner = InternalSourceScanner(artifacts, scorer.snapshot_dir / "tokenizer.json",
        profile.tokenizer_sha256, OriginalTaskSemanticChecker())
    admission = InternalDatasetAdmissionPolicy(artifacts, evaluation, profile.policy_ref,
        frozenset({profile.policy_ref}), scanner)
    candidate = DatasetCandidate(**manifest["candidate"])
    providers = {"original_task_registry"} | ({"approved_hub_registry"} if manifest["include_public"] else set())
    registry = DatasetProviderRegistry(provider_allowlist=providers)
    registry.register(OriginalTaskRegistryProvider((candidate,)))
    if manifest["include_public"]:
        registry.register(ApprovedHubRegistryProvider(("Salesforce/wikitext", "openai/gsm8k"), artifacts))
    intelligence = AutonomousDatasetIntelligence(registry,
        EvidenceBoundSourceInspector(artifacts, {candidate.candidate_id: manifest["review_ref"]}),
        admission, artifacts, DiscoveryBudget())
    state = CpuResearchState(repository, artifacts, workflow_id=spec.workflow_id, profile_ref=manifest["profile_ref"])
    return CpuResearchContext(root, artifacts, repository, admission, candidate,
        manifest["review_ref"], manifest["materialization_ref"], manifest["scope"], profile,
        manifest["profile_ref"], scorer.snapshot_dir, scorer, state,
        read_verified_json(artifacts, manifest["data_report_ref"]), intelligence,
        DatasetSearchRequest(**manifest["search_request"]), manifest["started_at"], spec), workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/hephaestus-model-cache"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config", type=Path, help="Immutable, policy-bounded campaign JSON; mandatory on new run")
    args = parser.parse_args()
    ctx, workspace = load_context(args.output.resolve(), cache_dir=args.cache_dir,
        resume=args.resume or args.verify_only, campaign_config_path=args.config)
    if args.verify_only:
        workflow = StagedWorkflowState.from_dict(ctx.repository.get_latest(WORKFLOW_STATE_COLLECTION, "workflow_id", ctx.scope["workflow_id"]))
    else:
        deps = compose_cpu_research_dependencies(ctx)
        workflow = build_staged_autonomous_orchestrator(**workspace.identity, dependencies=deps).run()
    admission = verify_campaign_workspace(workspace)
    summary = {"workflow_id": workflow.workflow_id, "status": workflow.status,
        "completed_operations": admission.verified_completed_prefix, "next_operation": admission.next_operation,
        "blocking_issues": workflow.blocking_issues, "external_certification": False}
    if workflow.completion_marker:
        summary["evidence"] = verify_cpu_research_trace(workflow, ctx)
    else:
        # Recover the original failure context without mutating or fabricating completion.
        from hephaestus.control.staged_state import StagedOperationRequest
        key = workflow.current_phase + "." + workflow.current_substep
        step = workflow.steps[key]
        if workflow.current_substep not in {"acquisition_approval_gate", "model_approval_gate", "action_approval_gate", "replay_evidence"}:
            # Diagnostic rerun is deliberately excluded: real science can have effects.
            summary["failed_operation_id"] = step.operation_id
    (args.output / "cpu_campaign_report.json").write_bytes(canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True))
    if not workflow.completion_marker:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
