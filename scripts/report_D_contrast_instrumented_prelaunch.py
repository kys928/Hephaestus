#!/usr/bin/env python3
"""Deterministic no-GPU prelaunch readiness report for D contrast proposal."""
from __future__ import annotations
import json
from pathlib import Path
import evaluator_contrast_inference_protocol as science
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

ROOT=Path(__file__).resolve().parents[1]
OUT=Path("D-contrast-instrumented-prelaunch.json")

def assemble():
    execution=json.loads((ROOT/"configs/experiments/evaluator_contrast_probe_execution.json").read_text())
    approval=json.loads((ROOT/"configs/experiments/evaluator_contrast_probe_approval.json").read_text())
    previous=json.loads((ROOT/"configs/experiments/hephaestus_evaluator_semantic_repair_v2.json").read_text())
    parent=json.loads((ROOT/previous["parent_candidate_manifest"]).read_text())
    source_manifest=science.manifest()
    unique,pairs=science.panel()
    still_pending=(execution.get("runpod_launch_enabled") is False and
                   approval.get("status")=="pending_explicit_user_launch_command" and
                   approval.get("additional_allocations_authorized") is False)
    newly_approved=(execution.get("runpod_launch_enabled") is True and
                    approval.get("status")=="authorized_once" and
                    approval.get("additional_allocations_authorized") is True)
    if (not (still_pending or newly_approved) or
        execution["approval_id"]!=approval["approval_id"] or
        approval["source_D_step96_sha256"]!=D_CHECKPOINT_SHA256 or
        source_manifest["unique_prompts"]!=56 or len(unique)!=56 or
        source_manifest["paired_challenges"]!=60 or len(pairs)!=60 or
        source_manifest["greedy_seed"]!=20261019):
        raise RuntimeError("unapproved launch or frozen scientific identity drift")
    if (execution["max_total_estimated_gpu_usd"]!=1.90 or
        execution["max_hourly_usd"]!=0.59 or
        execution["one_pod"] is not True or execution["extra_runs"] is not False or
        execution["runner_hard_wall_seconds"]>=execution["launcher_hard_wall_seconds"]):
        raise RuntimeError("bounded execution geometry drift")
    return dict(
        status=("PREPARED_BUT_NOT_AUTHORIZED" if still_pending else "EXPLICITLY_APPROVED_FOR_ONE_RUN"),
        training_performed_in_proposed_run=False,
        operation="inference-only, read-only model weights with diagnostic output to S3",
        scientific_protocol=source_manifest,
        model=dict(
            base=parent["model"]["model_id"],
            revision=parent["model"]["revision"],
            ancestor_parent_adapter_sha256=parent["model"]["adapter_sha256"],
            frozen_D_step96_adapter_sha256=D_CHECKPOINT_SHA256,
            trained_D_source_run_id=science.D_SOURCE_RUN,
            trained_D_source_repo_sha=science.D_SOURCE_SHA,
            precision=previous["model_protocol"]["precision"],
            LoRA_rank=previous["model_protocol"]["lora_rank"],
            LoRA_alpha=previous["model_protocol"]["lora_alpha"],
            load_kind=previous["roles"]["evaluator"]["load_kind"],
        ),
        already_completed_training_history=dict(
            status="historical source only; will not be rerun",
            selected_cases=previous["pack"]["selected_training_cases"],
            optimizer_steps=previous["training"]["maximum_optimizer_steps"],
            optimizer=previous["training"]["optimizer"],
            learning_rate=previous["training"]["learning_rate"],
            max_sequence_length=previous["training"]["max_sequence_length"],
            micro_batch_size=previous["training"]["micro_batch_size"],
            gradient_accumulation_steps=previous["training"]["gradient_accumulation_steps"],
            schedule_sha256=previous["arms"]["D"]["schedule_sha256"],
            training_pack_sha256=previous["arms"]["D"]["pack_sha256"],
        ),
        new_diagnostic_data=dict(
            source="newly authored fixed CPU oracle contrasts; no training split, no sealed cases",
            unique_model_prompts=len(unique),
            paired_challenges=len(pairs),
            appearances=source_manifest["case_appearances"],
            state_changing_pairs=sum(bool(x["scientific_change_expected"]) for x in pairs),
            invariant_pairs=sum(not bool(x["scientific_change_expected"]) for x in pairs),
            changed_factors=sorted({p["changed_factor"] for p in pairs}),
            hash_prompts=source_manifest["prompts_sha256"],
            hash_pairs=source_manifest["pairs_sha256"],
            max_new_tokens=execution["generation_max_new_tokens"],
            greedy_decoding=True,
            note="Identical prompts deduplicated. No targets appear in model input. Ground truth used only for CPU scoring."
        ),
        instrumentation=dict(
            schema=execution["durable_timing_schema"],
            persisted_per_stage_S3=True,
            checkpoints="source fetch, HF local-cache probe, ephemeral download, adapter integrity/extract, tokenizer, BF16 base weights, LoRA attachment, first generation, each 8-output shard",
            risk="Network-volume S3 timing writes incur latency; if evidence write fails the inference runner fails closed.",
        ),
        launch_boundary=dict(
            approved=newly_approved,
            new_explicit_user_command_required=still_pending,
            prior_gpu_approval_consumed=approval["prior_capacity_blocker"]["consumed_approval_id"],
            cloud_type="Secure",
            gpu_type=execution["gpu_type_id"],
            datacenter=execution["required_gpu_datacenter"],
            storage_datacenter=execution["storage_datacenter"],
            network_volume_mounted=False,
            network_volume_access="remote_S3_HTTPS",
            network_volume_id="cviwpryzao",
            quoted_usd_per_hour_max=execution["max_hourly_usd"],
            proposed_total_gpu_budget_usd=execution["max_total_estimated_gpu_usd"],
            per_run_allocations_max=1,
            max_wall_seconds=execution["launcher_hard_wall_seconds"],
            gpu_spend_authorized_now=(0 if still_pending else 1),
            note="Distinct user-approved A40 EU-SE-1 one-pod $1.90 cap with $0.59/h rate and EU-CZ-1 S3 backend; no guaranteed scientific completion."
        ),
        readiness_gates=dict(
            cpu_static_tests="pending workflow evidence",
            model_cache_warm="unverified; earlier S3 bootstrap only showed hf cache root, not revision complete",
            model_load_elapsed="unmeasured; earlier 15.5-minute pod left no post-load manifest",
            full_56_prompt_completion_under_cap="unproven; do not predict success or authorize budget automatically",
            action=("approved for one GPU allocation on a fresh launch-marker commit; independent verification required" if newly_approved else "await explicit user approval")
        ),
        forbidden_actions=["more than one new pod", "unbounded GPU charge", "training", "certification",
                          "promotion", "sealed-evidence reuse", "repeated allocation"],
    )

def main():
    record=assemble()
    OUT.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("D_CONTRAST_INSTRUMENTED_PRELAUNCH "+json.dumps({
        "status":record["status"],
        "training":record["training_performed_in_proposed_run"],
        "base":record["model"]["base"],
        "unique_prompts":record["new_diagnostic_data"]["unique_model_prompts"],
        "pairs":record["new_diagnostic_data"]["paired_challenges"],
        "new_approval_required":record["launch_boundary"]["new_explicit_user_command_required"],
        "gpu_spend_authorized_now":record["launch_boundary"]["gpu_spend_authorized_now"],
    },sort_keys=True))

if __name__=="__main__":
    main()
