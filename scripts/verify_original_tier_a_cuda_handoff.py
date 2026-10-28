#!/usr/bin/env python3
"""Revalidate archived original CPU 45/45 research before Tier A CUDA binding.

No GPU, S3 download, new scientific evaluation or model promotion.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from run_cpu_research_campaign import load_context
from hephaestus.control.staged_state import WORKFLOW_STATE_COLLECTION, StagedWorkflowState
from hephaestus.production.cpu_research import verify_cpu_research_trace
from hephaestus.production.tier_a_cuda_research import authenticate_original_tier_a_training
from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.storage.verified import canonical_bytes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-cpu-workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=Path("/tmp/hephaestus-model-cache"))
    parser.add_argument("--trusted-registry-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not replace immutable GPU source handoff receipt")
    if args.trusted_registry_sha256 != registry_digest():
        raise PermissionError("original five-model registry changed from operator trust root")
    ctx,workspace=load_context(args.original_cpu_workspace, cache_dir=args.model_cache,
                                resume=True,campaign_config_path=args.config)
    state=ctx.repository.get_latest(WORKFLOW_STATE_COLLECTION,"workflow_id",ctx.campaign_config.workflow_id)
    if state is None:
        raise PermissionError("original native research SQLite workflow is missing")
    workflow=StagedWorkflowState.from_dict(state)
    if not workflow.completion_marker:
        raise PermissionError("only authentic full CPU 45/45 campaign may enter handoff")
    evidence=verify_cpu_research_trace(workflow,ctx)
    if (evidence["verified_operations"]!=45
        or evidence["verified_domain_operations"]!=40
        or evidence["actual_training"]["completed_optimizer_steps"]!=96
        or evidence["claim_scope"]!="internal_prospective_generated_task_research"):
        raise PermissionError("native 45/45 Tier A scientific evidence incomplete")
    handoff=authenticate_original_tier_a_training(
        ctx,trusted_registry_sha256=args.trusted_registry_sha256)
    result={"schema":"original-tier-a-to-cuda-source-handoff.v1",
      "original_cpu_workflow_id":workflow.workflow_id,
      "original_cpu_native_trace_ref":evidence["native_trace_ref"],
      "source_and_corrected_recipe":asdict(handoff),
      "claim_scope":"internal_original_generated_task_research_only",
      "source_revalidated":True, "five_original_cuda_roles_loaded":False,
      "gpu_executed":False,"gpu_allocation_authorized":False,
      "independent_historical_certification":False,
      "issue_102_complete":False}
    args.output.write_bytes(canonical_bytes(result))
    print(json.dumps({"original_native_operations":45,
      "original_optimizer_steps":96,
      "source_ref":handoff.source_materialization_ref,
      "corrected_trainable_ref":handoff.transformed_training_ref,
      "gpu_executed":False},sort_keys=True))


if __name__=="__main__":
    main()
