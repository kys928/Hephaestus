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
from hephaestus.production.durable_campaign import verify_campaign_workspace
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
    # Verify original bytes BEFORE opening SQLite; read-only connections
    # can create transient -wal/-shm files not present in source archive.
    root=args.original_cpu_workspace.resolve()
    manifest=json.loads((root/"archive_sha256.json").read_bytes())
    actual={p.relative_to(root).as_posix():p for p in root.rglob("*")
            if p.is_file() and not p.is_symlink() and p.name!="archive_sha256.json"}
    if not isinstance(manifest,dict) or set(manifest)!=set(actual):
        missing=sorted(set(manifest)-set(actual))[:6] if isinstance(manifest,dict) else []
        unexpected=sorted(set(actual)-set(manifest))[:6] if isinstance(manifest,dict) else []
        raise PermissionError("original CPU artifact inventory missing or extra files: "
                              +str({"missing":missing,"unexpected":unexpected,
                                   "manifest_count":len(manifest) if isinstance(manifest,dict) else -1,
                                   "actual_count":len(actual)}))
    for name,expected in manifest.items():
        path=actual[name]
        digest=sha256()
        with path.open("rb") as stream:
            for piece in iter(lambda:stream.read(1024*1024),b""):
                digest.update(piece)
        if digest.hexdigest()!=expected:
            raise PermissionError("original CPU source member SHA mismatch")
    ctx,workspace=load_context(args.original_cpu_workspace, cache_dir=args.model_cache,
                                resume=True,campaign_config_path=args.config)
    state=ctx.repository.get_latest(WORKFLOW_STATE_COLLECTION,"workflow_id",ctx.campaign_config.workflow_id)
    if state is None:
        raise PermissionError("original native research SQLite workflow is missing")
    workflow=StagedWorkflowState.from_dict(state)
    if not workflow.completion_marker:
        raise PermissionError("only authentic full CPU 45/45 campaign may enter handoff")
    # A portable archive must never pretend the runner's original absolute
    # checkpoint paths are still live. Authenticate each original archived
    # member byte and the original SQLite receipts without rebinding history.
    state_check=verify_campaign_workspace(workspace)
    report=json.loads((root/"cpu_campaign_report.json").read_bytes())
    evidence=report.get("evidence") or {}
    training=evidence.get("actual_training") or {}
    if (not workflow.completion_marker
        or state_check.verified_completed_prefix!=45
        or report.get("status")!="completed"
        or report.get("completed_operations")!=45
        or evidence.get("verified_operations")!=45
        or evidence.get("verified_domain_operations")!=40
        or training.get("completed_optimizer_steps")!=96
        or training.get("full_dataset_epoch_completed") is not True
        or evidence.get("claim_scope")!="internal_prospective_generated_task_research"):
        raise PermissionError("native 45/45 Tier A original scientific evidence incomplete")
    handoff=authenticate_original_tier_a_training(
        ctx,trusted_registry_sha256=args.trusted_registry_sha256)
    result={"schema":"original-tier-a-to-cuda-source-handoff.v1",
      "original_cpu_workflow_id":workflow.workflow_id,
      "original_cpu_native_trace_ref":evidence["native_trace_ref"],
      "original_archive_members_sha256_checked":len(actual),
      "historical_checkpoint_paths_rebased":False,
      "resumable_checkpoint_recovery_claimed":False,
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
