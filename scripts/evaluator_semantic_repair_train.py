"""Continuous-optimizer training path for evaluator semantic-invariance repair."""
from __future__ import annotations
import hashlib,json,os,time,traceback
from pathlib import Path
from typing import Any,Mapping
import run_interface_repair_v1 as v1
import run_interface_repair_v2 as v2
from interface_repair_v2_bootstrap import materialize_parent_cached
import evaluator_semantic_repair_common as c

def run(cfg:Mapping[str,Any],parent:Mapping[str,Any],run_id:str)->int:
    import torch
    prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}";result_key=f'{prefix}/training/result.json';progress=f'{prefix}/training/progress.json';client=v1.s3_client();client.head_bucket(Bucket=v1.bucket());deadline=time.monotonic()+float(cfg['execution']['hard_wall_seconds_train']);repo=os.environ.get('HEPHAESTUS_REPO_SHA','').strip()
    if not repo:raise RuntimeError('HEPHAESTUS_REPO_SHA required')
    def hb(stage:str,**extra:Any)->None:
        p={'run_id':run_id,'stage':stage,'timestamp_unix':time.time(),'remaining_wall_seconds':max(0.,deadline-time.monotonic()),**extra};v1.put_json(client,progress,p);print('EVALUATOR_SEMANTIC_REPAIR_TRAIN_PROGRESS_JSON '+json.dumps(p,sort_keys=True),flush=True)
    result={'result_version':'hephaestus-evaluator-semantic-invariance-repair-training.v1','status':'running','run_id':run_id,'repo_sha':repo,'parent_candidate_id':parent['candidate_id'],'parent_adapter_sha256':parent['model']['adapter_sha256'],'training_performed':False,'development_diagnostic_touched':False,'burned_sealed_partition_used':False,'certification_claim_performed':False,'production_promotion_performed':False,'automatic_role_dispatch_performed':False}
    try:
        hb('curriculum_verification_started');pack,dsha,osha=c.pack_and_hashes(cfg);c.verify_baseline(client,cfg,dsha,osha);schedule,meta=c.training_schedule(pack,cfg);result.update(curriculum_sha256=cfg['pack']['canonical_sha256'],development_diagnostic_sha256=dsha,development_case_order_sha256=osha,training_selection=meta)
        v1.put_json(client,f'{prefix}/training/manifest.json',{'protocol':cfg,'run_id':run_id,'repo_sha':repo,'parent_candidate':{'candidate_id':parent['candidate_id'],'adapter_sha256':parent['model']['adapter_sha256'],'adapter_s3_key':parent['model']['adapter_s3_key']},'training_selection':meta,'development_case_bodies_included':False,'burned_sealed_case_bodies_included':False})
        spec=c.parent_spec(cfg,parent);root=Path('/opt/hephaestus-evaluator-semantic-invariance-repair')/run_id;root.mkdir(parents=True,exist_ok=True);hb('materializing_parent',adapter_sha256=spec['adapter']['sha256']);base,adapter=materialize_parent_cached(client,spec,root/'model');model,tok,runtime_info=v1.load_model(c.ROLE,cfg,spec,base,adapter);runtime=cfg['roles'][c.ROLE];trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n for n,_ in trainable):raise RuntimeError('only LoRA parameters may train')
        result.update(runtime=runtime_info,trainable_parameter_count=sum(p.numel() for _,p in trainable));hb('tokenization_preflight_started',selected_case_count=len(schedule));examples=[v2.training_example(tok,runtime,pack,x,c.ROLE,int(cfg['training']['max_sequence_length'])) for x in schedule];hb('tokenization_preflight_complete',maximum_nonpad_tokens=max(x['nonpad_tokens'] for x in examples),selected_case_order_sha256=meta['selected_case_order_sha256'])
        tr=cfg['training'];steps=int(tr['maximum_optimizer_steps']);accum=int(tr['gradient_accumulation_steps']);checkpoints=[int(x) for x in tr['checkpoint_steps']]
        if steps*accum!=len(examples) or checkpoints!=[16,32,48,64,80,96]:raise RuntimeError('optimizer/checkpoint geometry drift')
        if tr['gradient_checkpointing']:
            if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
            if hasattr(model,'gradient_checkpointing_enable'):model.gradient_checkpointing_enable()
        if hasattr(model.config,'use_cache'):model.config.use_cache=False
        opt=torch.optim.AdamW([p for _,p in trainable],lr=float(tr['learning_rate']),weight_decay=float(tr['weight_decay']));result.update(optimizer_constructed=True,optimizer_state_continuous_across_checkpoints=True);losses=[];supervised=nonpad=slots=cursor=0;artifacts=[];torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();model.train();hb('training_started',optimizer_steps=steps,gradient_accumulation_steps=accum)
        for step in range(1,steps+1):
            if time.monotonic()>=deadline:raise TimeoutError('training hard wall reached')
            opt.zero_grad(set_to_none=True);total=0.
            for _ in range(accum):
                ex=examples[cursor];cursor+=1;batch={k:v.to('cuda',non_blocking=True) for k,v in ex.items() if k in {'input_ids','attention_mask','labels'}};out=model(**batch,use_cache=False);loss=out.loss
                if not torch.isfinite(loss):raise RuntimeError(f'non-finite loss at step {step}')
                (loss/accum).backward();total+=float(loss.detach().cpu());supervised+=int(ex['supervised_tokens']);nonpad+=int(ex['nonpad_tokens']);slots+=int(ex['token_slots'])
            torch.nn.utils.clip_grad_norm_([p for _,p in trainable],float(tr['max_grad_norm']));opt.step();mean=total/accum;losses.append(mean)
            if step==1 or step%8==0:hb('training_progress',optimizer_step=step,optimizer_steps=steps,loss=mean,consumed_cases=cursor)
            if step in checkpoints:
                hb('checkpoint_persist_started',optimizer_step=step);art=v1.save_adapter(model,root/f'checkpoint-{step:03d}',client,f'{prefix}/checkpoints/step-{step:03d}/adapter.tar.gz');rec={'optimizer_step':step,'artifact':art,'loss_at_step':mean,'mean_loss_through_step':sum(losses)/len(losses),'consumed_training_cases':cursor,'consumed_case_id_sha256':hashlib.sha256('\n'.join(x['case_id'] for x in schedule[:cursor]).encode()).hexdigest()};artifacts.append(rec);v1.put_json(client,f'{prefix}/checkpoints/step-{step:03d}/training-checkpoint.json',rec);hb('checkpoint_persist_complete',optimizer_step=step,adapter_sha256=art['sha256'])
        torch.cuda.synchronize();secs=time.perf_counter()-started
        if cursor!=len(examples):raise RuntimeError('training schedule not fully consumed')
        result.update(status='training_complete',training_performed=True,backward_called=True,adapter_mutated=True,checkpoint_count=len(artifacts),checkpoint_artifacts=artifacts,training={'optimizer_steps':steps,'gradient_accumulation_steps':accum,'micro_batch_size':1,'learning_rate':float(tr['learning_rate']),'weight_decay':float(tr['weight_decay']),'selected_training_cases':len(schedule),'dynamic_token_slots':slots,'supervised_token_updates':supervised,'nonpad_token_updates':nonpad,'training_seconds':secs,'seconds_per_optimizer_step':secs/steps,'peak_training_vram_bytes':int(torch.cuda.max_memory_allocated()),'loss_first':losses[0],'loss_last':losses[-1],'loss_mean':sum(losses)/len(losses)},completed_at_unix=time.time());v1.put_json(client,result_key,result);hb('training_complete',checkpoint_count=len(artifacts),loss_last=losses[-1]);print('EVALUATOR_SEMANTIC_REPAIR_TRAIN_RESULT_JSON '+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed_at_unix=time.time())
        try:v1.put_json(client,result_key,result)
        except Exception:pass
        print('EVALUATOR_SEMANTIC_REPAIR_TRAIN_RESULT_JSON '+json.dumps(result,sort_keys=True),flush=True);raise
