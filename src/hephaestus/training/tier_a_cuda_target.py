"""Genuine CUDA-only target optimization on the original approved Tier A tokens.

A fresh campaign owns fresh checkpoints and receipts. No CPU execution fallback.
All 96 LoRA AdamW updates are persisted with original source indices and CUDA
device evidence; restart resumes from an authenticated optimizer/RNG snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re

from hephaestus.control.staged_state import StagedApprovalRequest, StagedApprovalDecision
from hephaestus.data.generated_factory import load_authorized_tokens
from hephaestus.policy.internal_research import put_record
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.training.cpu_chat_json_recipe import exact_chat_records
from hephaestus.training.cpu_research import hash_file, verify_snapshot
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.production.tier_a_cuda_research import (
    TierACudaDataHandoff, authenticate_original_tier_a_training,
)

_SHA=re.compile(r"sha256:[0-9a-f]{64}\Z")
TRAINING_RECIPE="chat_json_r8_full_epoch_96_v1"


@dataclass(frozen=True, slots=True)
class CudaTargetScope:
    """Externally fixed CUDA resource identity, separate from Tier A data."""
    workflow_id: str
    run_id: str
    trusted_registry_sha256: str
    trusted_approval_authority: str
    gpu_device: str = "cuda:0"
    maximum_optimizer_steps: int = 96
    maximum_gpu_seconds: int = 3600

    def validate(self):
        if (not self.workflow_id or not self.run_id
                or not self.trusted_approval_authority
                or self.gpu_device!="cuda:0"
                or not isinstance(self.trusted_registry_sha256,str)
                or not _SHA.fullmatch(self.trusted_registry_sha256)
                or type(self.maximum_optimizer_steps) is not int
                or self.maximum_optimizer_steps!=96
                or type(self.maximum_gpu_seconds) is not int
                or not 1<=self.maximum_gpu_seconds<=86400):
            raise PermissionError("explicit 96-step independent original CUDA resource scope required")
        return self


def verified_cuda_grant(ctx, operation_id, *, scope, service, purpose):
    """Only a separate external approval provider may authorize GPU resource use."""
    from hephaestus.roles.cpu_research_data import InternalResearchApprovalService
    scope.validate()
    if (type(service) is InternalResearchApprovalService
            or service is None
            or not callable(getattr(service,"decision_for",None))):
        raise PermissionError("Tier A data approval cannot authorize GPU execution")
    if (ctx.scope["workflow_id"],ctx.scope["run_id"])!=(scope.workflow_id,scope.run_id):
        raise PermissionError("GPU resource scope differs from native research campaign")
    if purpose not in ("target_training","target_evaluation"):
        raise PermissionError("unknown GPU scientific purpose")
    grant=StagedApprovalRequest(
        request_id="cuda-target:"+purpose+":"+operation_id,
        workflow_id=scope.workflow_id,run_id=scope.run_id,
        operation_id=operation_id,phase="resource_admission",
        substep=purpose,subject_ref=ctx.admission.policy_ref,
        requirements=("gpu_compute",purpose,"tier_a_original_research"),
    )
    decision=service.decision_for(grant)
    if (type(decision) is not StagedApprovalDecision
            or decision.request_id!=grant.request_id
            or decision.operation_id!=grant.operation_id
            or decision.subject_ref!=grant.subject_ref
            or decision.status!="approved"
            or set(decision.requirements)!=set(grant.requirements)
            or not _SHA.fullmatch(decision.approval_ref or "")):
        raise PermissionError("no independently authorized original CUDA target resource grant")
    record=read_verified_json(ctx.artifacts,decision.approval_ref)
    if (record.get("schema")!="original-cuda-resource-grant.v1"
            or record.get("workflow_id")!=scope.workflow_id
            or record.get("run_id")!=scope.run_id
            or record.get("operation_id")!=operation_id
            or record.get("source_policy_ref")!=ctx.admission.policy_ref
            or record.get("role")!=purpose
            or record.get("device")!="cuda:0"
            or record.get("status")!="approved"
            or record.get("authorized_by")!=scope.trusted_approval_authority
            or record.get("external_scientific_certification") is not False
            or type(record.get("maximum_gpu_seconds")) is not int
            or not 0<record["maximum_gpu_seconds"]<=scope.maximum_gpu_seconds):
        raise PermissionError("GPU grant identity, cost ceiling, or authority mismatch")
    return decision.approval_ref


def require_cuda_device(scope):
    import torch
    scope.validate()
    if not torch.cuda.is_available() or torch.cuda.device_count()<1:
        raise RuntimeError("CUDA execution unavailable; CPU substitution is forbidden")
    device=torch.device(scope.gpu_device)
    if device.type!="cuda" or device.index!=0:
        raise PermissionError("only explicitly pinned cuda:0 accepted")
    return device


def verify_cuda_target_checkpoint(receipt, *, expected_source=None, expected_scope=None):
    """Verify actual adapter + optimizer + CUDA RNG snapshot before accepting a step."""
    checkpoint=Path(receipt["checkpoint_ref"])
    if (receipt.get("schema") not in
            ("cuda-target-optimizer-step.v1","cuda-target-training-result.v1")
            or receipt.get("execution_device")!="cuda:0"
            or not validate_checkpoint_manifest(checkpoint)[0]
            or hash_file(checkpoint/"checkpoint_manifest.json")!=receipt.get("checkpoint_manifest_sha256")
            or hash_file(checkpoint/"adapter_model.safetensors")!=receipt.get("adapter_sha256")
            or (expected_source is not None and receipt.get("chat_trainable_ref")!=expected_source)
            or (expected_scope is not None and receipt.get("gpu_scope_ref")!=expected_scope)):
        raise PermissionError("CUDA checkpoint, immutable source or optimizer receipt differs")
    return checkpoint


class CudaExactTokenTrainer:
    def __init__(self, context, *, scope, grants, original_role_runtime=None):
        self.context=context
        self.scope=scope.validate()
        self.grants=grants
        self.original_role_runtime=original_role_runtime

    def run(self, operation_id):
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        ctx=self.context
        scope=self.scope
        grant_ref=verified_cuda_grant(ctx,operation_id,scope=scope,service=self.grants,
                                     purpose="target_training")
        device=require_cuda_device(scope)
        if self.original_role_runtime is not None:
            # The certified role generator retains one large model on CUDA.
            # Original role inference and target optimization must not coexist.
            self.original_role_runtime.close()
        ctx.check_authority(for_execution=False)
        handoff=authenticate_original_tier_a_training(
            ctx,trusted_registry_sha256=scope.trusted_registry_sha256)
        rows=load_authorized_tokens(
            ctx.admission,ctx.candidate,ctx.review_ref,ctx.materialization_ref,**ctx.scope)
        records,order,ref=exact_chat_records(ctx,rows)
        if (ref!=handoff.transformed_training_ref or len(order)!=96
                or len(records)!=96 or set(order)!=set(range(96))):
            raise PermissionError("CUDA trainer cannot use unapproved or partial training records")
        verify_snapshot(ctx.snapshot,ctx.profile.snapshot_hashes)
        scope_ref="sha256:"+sha256(canonical_bytes({
            "workflow_id":scope.workflow_id,"run_id":scope.run_id,
            "approved_device":scope.gpu_device,
            "source_policy_ref":handoff.source_policy_ref,
            "training_recipe":handoff.corrected_training_recipe_ref,
            "maximum_optimizer_steps":96})).hexdigest()
        key=scope_ref+":"+ref
        cached=ctx.state.get("cuda-train:"+operation_id,key)
        if cached:
            verify_cuda_target_checkpoint(cached,expected_source=ref,expected_scope=scope_ref)
            return cached
        torch.manual_seed(ctx.profile.seed)
        torch.cuda.manual_seed_all(ctx.profile.seed)
        model=AutoModelForCausalLM.from_pretrained(
            ctx.snapshot,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32).to(device)
        model=get_peft_model(model,LoraConfig(
            r=8,lora_alpha=16,lora_dropout=0.0,
            target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
        trainable=[p for p in model.parameters() if p.requires_grad]
        if not trainable or any(p.device.type!="cuda" for p in trainable):
            raise PermissionError("target optimizer parameters were not materialized on CUDA")
        optimizer=torch.optim.AdamW(trainable,lr=0.0005)
        root=ctx.root/"cuda-target-checkpoints"
        root.mkdir(exist_ok=True)
        rows_receipts=[r for r in ctx.repository.all("cuda_target_optimizer_steps")
                       if r.get("run_id")==scope.run_id]
        if len(rows_receipts)!=len({r.get("completed_steps") for r in rows_receipts}):
            raise PermissionError("duplicate CUDA optimizer receipt in original native state")
        if rows_receipts and set(r["completed_steps"] for r in rows_receipts)!=set(range(1,len(rows_receipts)+1)):
            raise PermissionError("CUDA optimizer history has gaps")
        # On restart, every earlier GPU receipt must still match its
        # immutable source/scope/adapter and exact original sample prefix;
        # a valid final checkpoint alone cannot conceal a forged gap.
        for expected_step, row in enumerate(rows_receipts, start=1):
            receipt=read_verified_json(ctx.artifacts,row["receipt_ref"])
            if (row.get("completed_steps")!=expected_step
                    or receipt.get("completed_steps")!=expected_step
                    or receipt.get("run_id")!=scope.run_id
                    or receipt.get("training_recipe")!=TRAINING_RECIPE
                    or receipt.get("consumed_sample_indices")!=order[:expected_step]):
                raise PermissionError("CUDA optimizer restart has nonoriginal step lineage")
            verify_cuda_target_checkpoint(
                receipt,expected_source=ref,expected_scope=scope_ref)
        completed=len(rows_receipts);losses=[];consumed=[]
        if completed:
            receipt=read_verified_json(ctx.artifacts,rows_receipts[-1]["receipt_ref"])
            prior=verify_cuda_target_checkpoint(receipt,expected_source=ref,expected_scope=scope_ref)
            from safetensors.torch import load_file
            weights=load_file(str(prior/"adapter_model.safetensors"))
            if set(weights)!=set(get_peft_model_state_dict(model)):
                raise PermissionError("resumed adapter keys changed")
            mismatch=set_peft_model_state_dict(model,weights)
            if mismatch.unexpected_keys:
                raise PermissionError("CUDA adapter resume contained unexpected keys")
            optimizer.load_state_dict(torch.load(prior/"optimizer.pt",map_location=device,weights_only=True))
            torch.set_rng_state(torch.load(prior/"rng_cpu.pt",weights_only=True))
            torch.cuda.set_rng_state(torch.load(prior/"rng_cuda0.pt",map_location="cpu",weights_only=True),device)
            losses,consumed=receipt["losses"],receipt["consumed_sample_indices"]
            if len(consumed)!=completed or consumed!=order[:completed]:
                raise PermissionError("resumed CUDA data sample order changed")
        started=__import__("time").monotonic()
        initial=[v.detach().clone() for v in trainable] if not completed else None
        for step in range(completed,96):
            if __import__("time").monotonic()-started>scope.maximum_gpu_seconds:
                raise PermissionError("independently bounded CUDA training resource grant expired")
            index=order[step];row=records[index]
            inputs=torch.tensor([row["input_ids"]],dtype=torch.long,device=device)
            labels=torch.tensor([row["labels"]],dtype=torch.long,device=device)
            if inputs.shape[1]>model.config.max_position_embeddings:
                raise ValueError("CUDA original task input would truncate")
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite genuine CUDA loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise ValueError("nonfinite genuine CUDA gradient")
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
            optimizer.step()
            torch.cuda.synchronize(device)
            losses.append(float(loss.detach().cpu()));consumed.append(index)
            target=root/f"step-{step+1}"
            staging=root/f"step-{step+1}.partial-{os.getpid()}"
            if staging.exists() or target.exists():
                raise FileExistsError("uncommitted CUDA partial/published checkpoint must be inspected")
            staging.mkdir()
            model.save_pretrained(staging,safe_serialization=True)
            torch.save(optimizer.state_dict(),staging/"optimizer.pt")
            torch.save(torch.get_rng_state(),staging/"rng_cpu.pt")
            torch.save(torch.cuda.get_rng_state(device),staging/"rng_cuda0.pt")
            components={p.name:hash_file(p) for p in sorted(staging.iterdir()) if p.is_file()}
            manifest={"schema":"checkpoint-manifest.v1","partial_write":False,
                "components":components,
                "manifest_hash":"sha256:"+sha256(canonical_bytes(components)).hexdigest()}
            (staging/"checkpoint_manifest.json").write_bytes(canonical_bytes(manifest))
            if not validate_checkpoint_manifest(staging)[0]:
                raise ValueError("CUDA optimizer/checkpoint manifest invalid")
            staging.rename(target)
            receipt={"schema":"cuda-target-optimizer-step.v1","run_id":scope.run_id,
                "operation_id":operation_id,"completed_steps":step+1,
                "execution_device":"cuda:0","cuda_runtime_version":torch.version.cuda,
                "training_recipe":TRAINING_RECIPE,
                "gpu_name":torch.cuda.get_device_name(device),
                "source_policy_ref":handoff.source_policy_ref,
                "chat_trainable_ref":ref,"gpu_scope_ref":scope_ref,
                "gpu_resource_approval_ref":grant_ref,
                "checkpoint_ref":str(target),"checkpoint_manifest_sha256":hash_file(target/"checkpoint_manifest.json"),
                "adapter_sha256":hash_file(target/"adapter_model.safetensors"),
                "losses":list(losses),"consumed_sample_indices":list(consumed),
                "gradient_norm":float(norm.detach().cpu()),
                "external_certification":False,"model_promoted":False}
            ctx.repository.append("cuda_target_optimizer_steps",{
                "run_id":scope.run_id,"completed_steps":step+1,
                "receipt_ref":put_record(ctx.artifacts,receipt)})
        final=read_verified_json(ctx.artifacts,
            ctx.repository.get_latest("cuda_target_optimizer_steps","run_id",scope.run_id)["receipt_ref"])
        checkpoint=verify_cuda_target_checkpoint(final,expected_source=ref,expected_scope=scope_ref)
        changed=completed>0 or any(
            not torch.equal(before,after.detach()) for before,after in zip(initial,trainable))
        if not changed or len(consumed)!=96 or set(consumed)!=set(range(96)):
            raise ValueError("CUDA optimizer failed to produce 96 actual complete ordered updates")
        report={"schema":"cuda-target-training-result.v1",
            "run_id":scope.run_id,"experiment_id":ctx.scope["experiment_id"],
            "backend_id":"actual_cuda_transformers_peft","execution_device":"cuda:0",
            "cuda_runtime_version":torch.version.cuda,"gpu_name":torch.cuda.get_device_name(device),
            "completed_optimizer_steps":96,"weights_changed":True,"losses":losses,
            "consumed_sample_indices":consumed,"source_rows_admitted":96,
            "training_recipe":TRAINING_RECIPE,"chat_trainable_ref":ref,
            "source_policy_ref":handoff.source_policy_ref,
            "gpu_scope_ref":scope_ref,"gpu_resource_approval_ref":grant_ref,
            "full_dataset_epoch_completed":True,"checkpoint_ref":str(checkpoint),
            "checkpoint_manifest_sha256":final["checkpoint_manifest_sha256"],
            "adapter_sha256":final["adapter_sha256"],
            "checkpoint_refs":[str(checkpoint)],"external_certification":False,
            "model_promoted":False,"status":"completed"}
        ctx.state.put("cuda-train:"+operation_id,key,report)
        return report
