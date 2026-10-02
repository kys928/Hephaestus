"""Immutable checkpoint evaluation path for evaluator semantic-invariance repair."""
from __future__ import annotations
import json,os,time,traceback
from pathlib import Path
from typing import Any,Mapping
import run_evaluator_sealed_certification_v1 as sealed
import run_evaluator_v3_1_development_baseline as baseline
import run_interface_repair_v1 as v1
import run_interface_repair_v3_1_evaluator as v31  # noqa: F401
from interface_repair_v2_bootstrap import materialize_parent_cached
import evaluator_semantic_repair_common as c

def artifact(training:Mapping[str,Any],step:int)->dict[str,Any]:
    for r in training.get('checkpoint_artifacts',[]):
        if int(r.get('optimizer_step',-1))==step and isinstance(r.get('artifact'),dict):return dict(r['artifact'])
    raise RuntimeError(f'checkpoint {step} artifact missing')

def run(cfg:Mapping[str,Any],parent:Mapping[str,Any],run_id:str,step:int)->int:
    if step not in [int(x) for x in cfg['training']['checkpoint_steps']]:raise RuntimeError('undeclared checkpoint step')
    prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}";ep=f'{prefix}/checkpoints/step-{step:03d}/evaluation';result_key=f'{ep}/result.json';progress=f'{ep}/progress.json';client=v1.s3_client();client.head_bucket(Bucket=v1.bucket());deadline=time.monotonic()+float(cfg['execution']['hard_wall_seconds_evaluate']);repo=os.environ.get('HEPHAESTUS_REPO_SHA','').strip()
    if not repo:raise RuntimeError('HEPHAESTUS_REPO_SHA required')
    def hb(stage:str,**extra:Any)->None:
        p={'run_id':run_id,'checkpoint_step':step,'stage':stage,'timestamp_unix':time.time(),'remaining_wall_seconds':max(0.,deadline-time.monotonic()),**extra};v1.put_json(client,progress,p);print('EVALUATOR_SEMANTIC_REPAIR_EVAL_PROGRESS_JSON '+json.dumps(p,sort_keys=True),flush=True)
    result={'result_version':'hephaestus-evaluator-semantic-invariance-repair-checkpoint-evaluation.v1','status':'running','run_id':run_id,'repo_sha':repo,'checkpoint_step':step,'training_performed_in_this_process':False,'weights_frozen_for_evaluation':True,'burned_sealed_partition_used':False,'certification_claim_performed':False,'production_promotion_performed':False}
    try:
        hb('checkpoint_verification_started');pack,dsha,osha=c.pack_and_hashes(cfg);base_result=c.verify_baseline(client,cfg,dsha,osha);training=c.s3_json(client,f'{prefix}/training/result.json')
        if training.get('status')!='training_complete' or training.get('repo_sha')!=repo or training.get('parent_adapter_sha256')!=cfg['required_parent_adapter_sha256']:raise RuntimeError('training identity mismatch')
        art=artifact(training,step);result.update(checkpoint_adapter=art,development_pack_sha256=cfg['pack']['canonical_sha256'],development_diagnostic_sha256=dsha,development_case_order_sha256=osha);spec=dict(c.parent_spec(cfg,parent));spec['adapter']={'s3_key':art['s3_key'],'sha256':art['sha256'],'bytes':int(art['bytes'])};spec['status']=f'semantic_invariance_repair_checkpoint_{step:03d}';root=Path('/opt/hephaestus-evaluator-semantic-invariance-eval')/run_id/f'step-{step:03d}';root.mkdir(parents=True,exist_ok=True);hb('materializing_checkpoint',adapter_sha256=art['sha256']);base,adapter=materialize_parent_cached(client,spec,root/'model');model,tok,runtime_info=v1.load_model(c.ROLE,cfg,spec,base,adapter)
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
        if sum(p.numel() for p in model.parameters() if p.requires_grad):raise RuntimeError('evaluation model not frozen')
        pf_rows,pf=sealed.evaluate_partition(cfg=cfg,pack=pack,partition='preflight',model=model,tokenizer=tok,deadline=deadline,client=client,prefix=ep,heartbeat=hb);rows,summary=sealed.evaluate_partition(cfg=cfg,pack=pack,partition='development_diagnostic',model=model,tokenizer=tok,deadline=deadline,client=client,prefix=ep,heartbeat=hb);analysis=sealed.decision_analysis(rows);gates=baseline.development_gates(summary,analysis);bs=base_result['development_summary']
        result.update(status='evaluation_complete',runtime={**runtime_info,'trainable_parameter_count_after_freeze':0},preflight={'sample_count':len(pf_rows),'summary':pf},development_summary=summary,development_analysis=analysis,checkpoint_selection_gates=gates,baseline_comparison={'baseline_run_id':cfg['baseline']['run_id'],'baseline_decision_exact_pass_rate':bs['decision_exact_pass_rate'],'checkpoint_decision_exact_pass_rate':summary['decision_exact_pass_rate'],'decision_exact_gain':float(summary['decision_exact_pass_rate'])-float(bs['decision_exact_pass_rate']),'baseline_semantic_exact_pass_rate':bs['semantic_exact_pass_rate'],'checkpoint_semantic_exact_pass_rate':summary['semantic_exact_pass_rate'],'semantic_exact_gain':float(summary['semantic_exact_pass_rate'])-float(bs['semantic_exact_pass_rate']),'baseline_per_state_decision_accuracy':base_result['development_analysis']['per_state_decision_accuracy']},completed_at_unix=time.time());v1.put_json(client,result_key,result);hb('evaluation_complete',gate_passed=gates['passed'],decision_exact=summary['decision_exact_pass_rate'],semantic_exact=summary['semantic_exact_pass_rate']);print('EVALUATOR_SEMANTIC_REPAIR_EVAL_RESULT_JSON '+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed_at_unix=time.time())
        try:v1.put_json(client,result_key,result)
        except Exception:pass
        print('EVALUATOR_SEMANTIC_REPAIR_EVAL_RESULT_JSON '+json.dumps(result,sort_keys=True),flush=True);raise
