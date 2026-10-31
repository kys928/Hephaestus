"""Actual CUDA:0 original-task greedy generation, NLL, and exact JSON outcomes.

Training/evaluation are separate from the five original certified CUDA
advisory-role model identities. No CPU fallback or certification shortcut.
"""
from .exact_json import SCORER_VERSION, score_prediction

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from hephaestus.data.generated_validation import validate_original_example
from hephaestus.evaluation.cpu_prospective import CpuProspectiveEvaluator
from hephaestus.policy.internal_research import put_record
from hephaestus.storage.verified import read_verified_json
from hephaestus.training.cpu_chat_json_recipe import chat_prompt
from hephaestus.training.tier_a_cuda_target import (
    require_cuda_device,verified_cuda_grant,verify_cuda_target_checkpoint,
)


class CudaProspectiveEvaluator:
    """Real original generated-task CUDA evaluation with no answer postprocessing."""
    def __init__(self,context, *, scope, grants, original_role_runtime=None):
        self.ctx=context
        self.scope=scope
        self.grants=grants
        self.original_role_runtime=original_role_runtime

    def generate(self,operation_id,*,checkpoint=None,repeat=0):
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        ctx=self.ctx
        grant_ref=verified_cuda_grant(ctx,operation_id,scope=self.scope,
            service=self.grants,purpose="target_evaluation")
        import time
        started=time.monotonic()
        ceiling=min(self.scope.maximum_gpu_seconds,
            read_verified_json(ctx.artifacts,grant_ref)["maximum_gpu_seconds"])
        device=require_cuda_device(self.scope)
        if self.original_role_runtime is not None:
            self.original_role_runtime.close()
        ctx.check_authority(for_execution=False)
        policy=ctx.admission.evaluation_store.require_frozen(
            ctx.scope["experiment_id"],ctx.profile_protocol)
        if policy.get("historical_reference") is not False:
            raise PermissionError("historical protected evaluation is not part of Tier A")
        input_ref=put_record(ctx.artifacts,{"schema":"cuda-target-evaluation-request.v2", "scorer_version":SCORER_VERSION,
            "protocol_ref":ctx.profile_protocol,"gpu_scope":self.scope.run_id,
            "gpu_grant_ref":grant_ref,"operation_id":operation_id,
            "checkpoint":str(checkpoint) if checkpoint else "pinned-pretrained-baseline",
            "repeat":repeat,"profile_ref":ctx.profile_ref})
        if checkpoint is not None:
            # Revalidate original optimizer receipt rather than computing an
            # expected hash from the candidate's mutable present-day bytes.
            from hephaestus.production.tier_a_cuda_research import authenticate_original_tier_a_training
            handoff=authenticate_original_tier_a_training(
                ctx,trusted_registry_sha256=self.scope.trusted_registry_sha256)
            latest=ctx.repository.get_latest(
                "cuda_target_optimizer_steps","run_id",self.scope.run_id)
            if latest is None or latest.get("completed_steps")!=96:
                raise PermissionError("CUDA candidate has no complete original optimizer history")
            receipt=read_verified_json(ctx.artifacts,latest["receipt_ref"])
            if receipt.get("checkpoint_ref")!=str(checkpoint):
                raise PermissionError("CUDA candidate checkpoint is not the original finalized step")
            verify_cuda_target_checkpoint(
                receipt,expected_source=handoff.transformed_training_ref)
        cached=ctx.state.get("cuda-eval:"+operation_id,input_ref)
        if cached:
            if cached.get("execution_device")!="cuda:0":
                raise PermissionError("cached CUDA evaluation cannot claim CPU generation")
            return cached
        rows=read_verified_json(ctx.artifacts,policy["evaluation_ref"])["examples"]
        if len(rows)!=24:
            raise PermissionError("original Tier A evaluation protocol must retain 24 fixed examples")
        snapshot=ctx.snapshot
        from hephaestus.training.cpu_research import verify_snapshot
        verify_snapshot(snapshot,ctx.profile.snapshot_hashes)
        model=AutoModelForCausalLM.from_pretrained(
            snapshot,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32).to(device)
        if checkpoint is not None:
            from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
            if not validate_checkpoint_manifest(Path(checkpoint))[0]:
                raise PermissionError("actual CUDA candidate checkpoint invalid")
            model=PeftModel.from_pretrained(
                model,checkpoint,is_trainable=False,local_files_only=True).to(device)
        if next(model.parameters()).device.type!="cuda":
            raise PermissionError("real generation must execute on CUDA")
        model.eval()
        torch.manual_seed(ctx.profile.seed)
        torch.cuda.manual_seed_all(ctx.profile.seed)
        tokenizer=ctx.scorer.tokenizer
        outputs=[]
        for row in rows:
            if time.monotonic()-started>ceiling:
                raise PermissionError("actual CUDA evaluation exceeded approved GPU resource budget")
            validate_original_example(row)
            prefix=chat_prompt(tokenizer,row["prompt"])
            tokens=tokenizer(prefix,return_tensors="pt",add_special_tokens=False)
            ids=tokens.input_ids[0].tolist()
            target_ids=tokenizer(row["target"],add_special_tokens=False).input_ids
            if not target_ids or len(ids)+max(64,len(target_ids))>model.config.max_position_embeddings:
                raise ValueError("original frozen evaluation cannot truncate")
            cuda_input={k:v.to(device) for k,v in tokens.items()}
            combined=torch.tensor([ids+target_ids],dtype=torch.long,device=device)
            labels=torch.tensor([[-100]*len(ids)+target_ids],dtype=torch.long,device=device)
            with torch.inference_mode():
                generated=model.generate(**cuda_input,max_new_tokens=64,do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
                nll=model(input_ids=combined,attention_mask=torch.ones_like(combined),
                          labels=labels).loss
            if not bool(torch.isfinite(nll)):
                raise ValueError("nonfinite actual CUDA evaluation target NLL")
            answer_text=tokenizer.decode(
                generated[0,tokens.input_ids.shape[1]:].detach().cpu(),
                skip_special_tokens=True).strip()
            valid=correct=False
            measurement=score_prediction(answer_text,row["target"])
            valid,correct=measurement["valid_json"],measurement["exact_correct"]
            outputs.append({"sample_id":row["sample_id"],"family":row["family"],
                "prompt_sha256":"sha256:"+sha256(prefix.encode()).hexdigest(),
                "output":answer_text,
                "output_sha256":"sha256:"+sha256(answer_text.encode()).hexdigest(),
                "json_valid":valid,"correct":correct,
                "target_nll":float(nll.detach().cpu())})
        torch.cuda.synchronize(device)
        result={"schema":"cuda-target-prospective-real-generation.v2", "scorer_version":SCORER_VERSION,
            "profile_ref":ctx.profile_ref,"protocol_ref":ctx.profile_protocol,
            "checkpoint_ref":str(checkpoint) if checkpoint else None,
            "generation_settings_id":"greedy-chat64-json-eos-no-truncation.v2",
            "execution_device":"cuda:0","cuda_runtime_version":torch.version.cuda,
            "gpu_name":torch.cuda.get_device_name(device),
            "gpu_resource_approval_ref":grant_ref,"gpu_scope_run_id":self.scope.run_id,
            "samples":outputs,"correct":sum(r["correct"] for r in outputs),
            "json_valid":sum(r["json_valid"] for r in outputs),"count":len(outputs),
            "family_counts":dict(Counter(r["family"] for r in outputs)),
            "mean_teacher_forced_target_nll":sum(r["target_nll"] for r in outputs)/len(outputs),
            "external_certification":False,"independent_holdout":False,"repeat_index":repeat}
        ctx.state.put("cuda-eval:"+operation_id,input_ref,result)
        del model
        torch.cuda.empty_cache()
        return result

    def compare(self,baseline,candidate):
        if (baseline.get("execution_device")!="cuda:0" or
            candidate.get("execution_device")!="cuda:0"):
            raise PermissionError("real baseline and candidate must both execute on CUDA")
        return CpuProspectiveEvaluator(self.ctx).compare(baseline,candidate)
