#!/usr/bin/env python3
"""Explicit V2 training path, gated off until a new paid-experiment approval.

Uses the existing frozen Granite/PEFT loader and S3 materializer. No production
boundary, frozen V1 runner or parent adapter is changed. Call train with an
externally approved scientific config; the checked-in config refuses paid work.
"""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
import build_evaluator_semantic_repair_v2 as b
import run_interface_repair_v1 as runtime
from interface_repair_v2_bootstrap import materialize_parent_cached
from evaluator_semantic_v2_supervision import training_example
from prepare_evaluator_semantic_repair_v2 import prepare


def authorize(cfg, manifest):
    if not (cfg['governance'].get('training_approved') is True and cfg['governance'].get('paid_training_allowed') is True):
        raise PermissionError('V2 paid training is not approved; frozen V1/recovery approval does not authorize it')
    if os.environ.get('HEPHAESTUS_EVALUATOR_V2_AUTHORIZED') != 'LAUNCH_APPROVED_EVALUATOR_SEMANTIC_V2':
        raise PermissionError('V2 explicit execution authorization is missing')
    if manifest['arm'] == 'B':
        raise PermissionError('B is a CPU equivalence audit; do not buy duplicate training')
    for k in ('parent_adapter_mutation_allowed','burned_sealed_partition_use_allowed','certification_claim_allowed','production_promotion_allowed','automatic_role_dispatch_allowed'):
        if cfg['governance'].get(k) is not False:
            raise ValueError(f'forbidden authority: {k}')
    if b.digest(cfg) != manifest['scientific_config_sha256']:
        raise ValueError('approved config/manifest mismatch')


def train(cfg, parent, pack, selected, manifest, *, client, run_id, repo_sha, root):
    authorize(cfg, manifest)
    import torch
    if not repo_sha or len(repo_sha) != 40 or not run_id or any(x in run_id for x in ('/','..')):
        raise ValueError('immutable repo identity and safe run ID required')
    if (parent['model']['revision'] != cfg['required_base_revision'] or
        parent['model']['adapter_sha256'] != cfg['required_parent_adapter_sha256'] or
        b.digest(selected) != manifest['selected_schedule_sha256']):
        raise ValueError('lineage or schedule drift')
    prefix = cfg['execution']['s3_prefix'] + '/' + run_id
    # Refuse overwrite/replay: caller must use a fresh run and output root.
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + cfg['execution']['hard_wall_seconds_train']
    random.seed(cfg['training']['seed'])
    torch.manual_seed(cfg['training']['seed'])
    torch.cuda.manual_seed_all(cfg['training']['seed'])
    spec = b.common.parent_spec(cfg,parent)
    base, adapter = materialize_parent_cached(client,spec,root/'model')
    model,tok,info = runtime.load_model('evaluator',cfg,spec,base,adapter)
    if info['revision'] != cfg['required_base_revision']:
        raise ValueError('loaded base revision mismatch')
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not params or any('lora_' not in n for n,p in params):
        raise ValueError('only parent LoRA parameters may train')
    # Preserve the pinned PEFT loader: it may upcast trainable adapter weights
    # to FP32 while base weights remain BF16. Record rather than silently recast.
    base_dtypes = {str(p.dtype) for n,p in model.named_parameters() if 'lora_' not in n and p.is_floating_point()}
    if base_dtypes != {'torch.bfloat16'}:
        raise ValueError('frozen base precision is not BF16')
    info.update(base_parameter_dtypes=sorted(base_dtypes),
                adapter_parameter_dtypes=sorted({str(p.dtype) for _,p in params}))
    tr=cfg['training']
    examples=[training_example(tok,cfg['roles']['evaluator'],pack,r,'evaluator',tr['max_sequence_length']) for r in selected]
    if len(examples) != tr['maximum_optimizer_steps']*tr['gradient_accumulation_steps']:
        raise ValueError('training dose drift')
    if tr['gradient_checkpointing']:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for _,p in params],lr=tr['learning_rate'],weight_decay=tr['weight_decay'])
    runtime.put_json(client,prefix+'/training/manifest.json',{**manifest,'repo_sha':repo_sha,'run_id':run_id,'runtime':info})
    model.train()
    cursor=0
    artifacts=[]
    losses=[]
    for step in range(1,tr['maximum_optimizer_steps']+1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum=0.
        for _ in range(tr['gradient_accumulation_steps']):
            if time.monotonic() >= deadline:
                raise TimeoutError('V2 training hard wall reached')
            ex=examples[cursor];cursor+=1
            batch={k:ex[k].to('cuda') for k in ('input_ids','attention_mask','labels')}
            loss=model(**batch,use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            (loss/tr['gradient_accumulation_steps']).backward()
            loss_sum+=float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_([p for _,p in params],tr['max_grad_norm'])
        optimizer.step()
        losses.append(loss_sum/tr['gradient_accumulation_steps'])
        if step in tr['checkpoint_steps']:
            art=runtime.save_adapter(model,root/f'checkpoint-{step:03d}',client,f'{prefix}/checkpoints/step-{step:03d}/adapter.tar.gz')
            artifacts.append(dict(optimizer_step=step,artifact=art,consumed_cases=cursor))
        runtime.put_json(client,prefix+'/training/progress.json',dict(optimizer_step=step,loss=losses[-1],consumed_cases=cursor))
    result=dict(status='training_complete',repo_sha=repo_sha,run_id=run_id,
        parent_adapter_sha256=cfg['required_parent_adapter_sha256'],base_revision=cfg['required_base_revision'],
        checkpoint_artifacts=artifacts,training_manifest_sha256=b.digest(manifest),
        optimizer_state_continuous_across_checkpoints=True,losses=losses,
        scientific_improvement_demonstrated=False,certification_claim=False)
    runtime.put_json(client,prefix+'/training/result.json',result)
    return result


if __name__ == '__main__':
    # Preparation and evaluation are separate from paid resource orchestration.
    # Default path stops before importing torch, contacting S3 or loading weights.
    cfg, pack, selected, manifest=prepare('C')
    authorize(cfg,manifest)
    raise RuntimeError('Use a reviewed governed launcher with hard cost ceilings and verified teardown')
