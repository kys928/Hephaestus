"""Opt-in real one-GPU Prometheus campaign worker, strictly decision-first.

Provider lifecycle and cost monitoring are external to this process. The worker
may only run in an independently admitted, supervised paid RunPod allocation.
It never issues API pod-create/delete calls, modifies frozen weights or opens a
protected final assessment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from hephaestus.development.prometheus_scientific_decision import (
    _verify, PARENT_SHA, D_SHA, decide, prepare,
)
from hephaestus.development.prometheus_granite_public_measurement import run_public_measurement
from hephaestus.development.prometheus_global_adapter_source import VerifiedGlobalRoleArchiveSource
from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from hephaestus.llm.role_contract import RoleMasteryContract
from hephaestus.providers.models.role_stack import load_certified_role_model_stack
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--d-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--s3-endpoint")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-region", default="EU-CZ-1")
    parser.add_argument("--global-volume-root", type=Path,
                        help="Opt-in read-only mounted Global archive source")
    parser.add_argument("--global-volume-id",
                        help="Externally verified exact mounted provider volume ID")
    parser.add_argument("--adapter-cache", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--registry", type=Path,
        default=Path("configs/models/hephaestus_role_model_stack_v1.json"))
    parser.add_argument("--contract", type=Path,
        default=Path("configs/models/hephaestus_role_mastery_runtime_v1.json"))
    args = parser.parse_args(argv)
    parent = _verify(args.parent_report, PARENT_SHA)
    d = _verify(args.d_report, D_SHA)
    prepare(parent, d)  # Source/ontology/authority preflight BEFORE GPU import.
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Prometheus GPU campaign output must be a fresh immutable directory")
    stack = load_certified_role_model_stack(args.registry)
    if args.global_volume_root is not None:
        if not args.global_volume_id or args.s3_endpoint or args.s3_bucket:
            raise ValueError("Global mode requires exact volume ID and excludes S3 adapter fallback")
        root = args.global_volume_root.resolve(strict=True)
        for write_root in (args.out, args.adapter_cache, args.hf_cache):
            if write_root.resolve().is_relative_to(root):
                raise PermissionError("Prometheus mutable output/cache cannot live on Global object storage")
        adapter_client = VerifiedGlobalRoleArchiveSource(
            args.global_volume_root, args.global_volume_id, stack)
        adapter_bucket = args.global_volume_id
    else:
        if not args.s3_endpoint or not args.s3_bucket or args.global_volume_id:
            raise ValueError("Regional mode requires complete S3 endpoint/bucket without Global mount")
        import boto3
        adapter_client = boto3.client(
            "s3", endpoint_url=args.s3_endpoint, region_name=args.s3_region)
        adapter_bucket = args.s3_bucket
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"campaign-precommit.json").write_bytes(canonical_bytes({
        "schema":"prometheus-real-gpu-campaign-precommit.v1",
        "parent_evidence_sha256":"sha256:"+PARENT_SHA,
        "historical_D_evidence_sha256":"sha256:"+D_SHA,
        "original_parent_unchanged":True,
        "first_scientific_action_must_come_from_hephaestus":True,
        "new_gpu_model_inference_completed_at_precommit":False,
        "training_steps_permitted":0,
        "production_role_activation_permitted":False,
        "independent_final_holdout_access_permitted":False,
    }))
    generator = PinnedTransformersRoleGenerator(
        args.hf_cache, device="cuda:0", allow_remote_pinned_download=True)
    reasoner = CertifiedRoleReasoner(
        stack=stack,
        contract=RoleMasteryContract.load(args.contract),
        adapters=VerifiedRoleAdapters(
            adapter_client, adapter_bucket, args.adapter_cache),
        generator=generator,
        artifact_store=FileSystemArtifactStore(args.out/"decision"/"inference-receipts"),
    )
    status="not_started"
    try:
        decision = decide(parent,d,reasoner=reasoner,out=args.out/"decision")
        chosen = decision["independently_admitted_intervention"]["intervention_kind"]
        if chosen == "stop":
            status="scientist_selected_stop"
        elif chosen == "collect_more_evidence":
            # The MODEL chooses the admitted scientific action. The specific
            # public instrument is mapped by deterministic implementation
            # policy; never claim the model selected its identity.
            measurement=run_public_measurement(
                reasoner,decision_root=args.out/"decision",
                output=args.out/"public-sixstate-diagnostic")
            status="development_measurement_completed"
        else:
            status="scientist_selected_unsupported_execution"
        result={
            "schema":"prometheus-actual-gpu-cycle-result.v1",
            "status":status,
            "scientist_decision_owner":"HEPHAESTUS_ACTUAL_FROZEN_DIAGNOSIS_AND_PLANNER",
            "tool_selection_owner":"DETERMINISTIC_EXECUTION_POLICY",
            "scientific_action":chosen,
            "model_inference_executed":True,
            "public_development_measurement_executed":status=="development_measurement_completed",
            "reference_model_preserved":True,
            "new_candidate_trained":False,
            "independent_certification":False,
            "production_promotion":False,
            "subsequent_research_decision_required":status=="development_measurement_completed",
        }
        (args.out/"campaign-result.json").write_bytes(canonical_bytes(result))
        print("PROMETHEUS_REAL_GPU_CAMPAIGN "+json.dumps(result,sort_keys=True),flush=True)
        return 0 if status!="scientist_selected_unsupported_execution" else 3
    except Exception as exc:
        (args.out/"failure.json").write_bytes(canonical_bytes({
            "schema":"prometheus-incomplete-gpu-cycle.v1",
            "error_class":type(exc).__name__,
            "status":"failed_incomplete_no_claim",
            "model_output_cannot_be_inferred":True,
            "not_an_autonomous_scientific_success":True,
        }))
        raise
    finally:
        generator.close()


if __name__ == "__main__":
    raise SystemExit(main())
