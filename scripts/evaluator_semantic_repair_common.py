"""Shared protocol guards and deterministic curriculum scheduling for evaluator repair."""
from __future__ import annotations
import hashlib, json, random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
import build_evaluator_semantic_invariance_repair_v1 as curriculum
import run_evaluator_sealed_certification_v1 as sealed
import run_interface_repair_v1 as v1

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/'configs/experiments/hephaestus_evaluator_semantic_invariance_repair_v1.json'
ROLE='evaluator'
FAMILIES=('factorial_generalization','minimal_contrast','precedence_and_field_isolation')

def jload(path:Path)->dict[str,Any]:
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise RuntimeError(f'{path} must contain a JSON object')
    return x

def cfg_and_parent()->tuple[dict[str,Any],dict[str,Any]]:
    cfg=jload(CFG_PATH); g=cfg['governance']
    if not(g.get('training_approved') and g.get('paid_training_allowed')): raise RuntimeError('repair is not approved')
    if g.get('parent_adapter_mutation_allowed') is not False or g.get('burned_sealed_partition_use_allowed') is not False: raise RuntimeError('repair cleanliness governance drift')
    if g.get('certification_claim_allowed') or g.get('production_promotion_allowed') or g.get('automatic_role_dispatch_allowed'): raise RuntimeError('repair may not certify, promote, or dispatch')
    p=sealed.load_json(ROOT/cfg['parent_candidate_manifest'])
    if p.get('candidate_id')!=cfg['required_parent_candidate_id']: raise RuntimeError('parent candidate drift')
    if p.get('model',{}).get('adapter_sha256')!=cfg['required_parent_adapter_sha256']: raise RuntimeError('parent adapter drift')
    if p.get('governance',{}).get('weights_frozen') is not True: raise RuntimeError('parent is not frozen')
    return cfg,p

def pack_and_hashes(cfg:Mapping[str,Any])->tuple[dict[str,Any],str,str]:
    pack=curriculum.build_pack(); curriculum.validate(pack)
    if curriculum.canonical_sha256(pack)!=cfg['pack']['canonical_sha256']: raise RuntimeError('curriculum hash drift')
    dev=list(pack['partitions'][ROLE]['development_diagnostic']); raw=json.dumps(dev,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode(); dsha=hashlib.sha256(raw).hexdigest(); osha=hashlib.sha256('\n'.join(str(x['case_id']) for x in dev).encode()).hexdigest()
    if dsha!=cfg['pack']['development_diagnostic_sha256'] or osha!=cfg['pack']['development_case_order_sha256']: raise RuntimeError('development diagnostic identity drift')
    return pack,dsha,osha

def s3_json(client:Any,key:str)->dict[str,Any]:
    r=client.get_object(Bucket=v1.bucket(),Key=key)
    try:x=json.loads(r['Body'].read().decode())
    finally:r['Body'].close()
    if not isinstance(x,dict):raise RuntimeError(f'S3 object is not JSON: {key}')
    return x

def verify_baseline(client:Any,cfg:Mapping[str,Any],dsha:str,osha:str)->dict[str,Any]:
    b=s3_json(client,cfg['baseline']['s3_key']); s=b.get('development_summary',{})
    ok=(b.get('status')=='baseline_complete' and b.get('candidate_adapter_sha256')==cfg['required_parent_adapter_sha256'] and b.get('development_pack_sha256')==cfg['pack']['canonical_sha256'] and b.get('development_diagnostic_sha256')==dsha and b.get('development_case_order_sha256')==osha and abs(float(s.get('decision_exact_pass_rate',-1))-float(cfg['baseline']['decision_exact_pass_rate']))<1e-12)
    if not ok:raise RuntimeError('frozen development baseline identity drift')
    return b

def stable(rows:list[dict[str,Any]],seed:str)->list[dict[str,Any]]:
    return sorted(rows,key=lambda x:hashlib.sha256(f"{seed}:{x['case_id']}".encode()).hexdigest())

def upstream_correct(r:Mapping[str,Any])->bool:
    u=r.get('upstream_output'); return isinstance(u,Mapping) and str(u.get('action'))==str(r['expected']['action'])

def training_schedule(pack:Mapping[str,Any],cfg:Mapping[str,Any])->tuple[list[dict[str,Any]],dict[str,Any]]:
    train=list(pack['partitions'][ROLE]['train']); seed=int(cfg['training']['seed']); selected={}
    for state in curriculum.STATE_ORDER:
        rows=[dict(x) for x in train if x['semantic_root']==state]; fam={f:[x for x in rows if x['curriculum_family']==f] for f in FAMILIES}
        if len(fam['minimal_contrast'])!=64:raise RuntimeError(f'{state} contrast count drift')
        chosen=list(fam['minimal_contrast'])
        for f in ('factorial_generalization','precedence_and_field_isolation'):
            direct=stable([x for x in fam[f] if x['kind']=='rehearsal'],f'{seed}:{state}:{f}:d'); hand=[x for x in fam[f] if x['kind']=='interface']; good=stable([x for x in hand if upstream_correct(x)],f'{seed}:{state}:{f}:g'); bad=stable([x for x in hand if not upstream_correct(x)],f'{seed}:{state}:{f}:b')
            if len(direct)<8 or len(good)<4 or len(bad)<4:raise RuntimeError(f'{state}/{f} balance unavailable')
            chosen+=direct[:8]+good[:4]+bad[:4]
        if Counter(x['curriculum_family'] for x in chosen)!=Counter({'minimal_contrast':64,'factorial_generalization':16,'precedence_and_field_isolation':16}):raise RuntimeError(f'{state} family drift')
        if Counter(x['kind'] for x in chosen)!=Counter({'interface':48,'rehearsal':48}):raise RuntimeError(f'{state} handoff drift')
        selected[state]=chosen
    sizes={'minimal_contrast':[10,11,11,10,11,11],'factorial_generalization':[3,3,2,3,3,2],'precedence_and_field_isolation':[3,2,3,3,2,3]}; blocks=[[] for _ in range(6)]
    for state in curriculum.STATE_ORDER:
        for f in FAMILIES:
            ordered=stable([x for x in selected[state] if x['curriculum_family']==f],f'{seed}:{state}:{f}:schedule'); c=0
            for i,n in enumerate(sizes[f]):blocks[i]+=ordered[c:c+n];c+=n
            if c!=len(ordered):raise RuntimeError('schedule allocation drift')
    schedule=[]; summaries=[]
    for i,b in enumerate(blocks):
        if len(b)!=128 or Counter(x['semantic_root'] for x in b)!=Counter({s:16 for s in curriculum.STATE_ORDER}):raise RuntimeError('checkpoint block balance drift')
        random.Random(seed+1000+i).shuffle(b);schedule+=b;summaries.append({'block':i+1,'optimizer_steps_through':16*(i+1),'case_count':128,'state_counts':dict(sorted(Counter(x['semantic_root'] for x in b).items())),'family_counts':dict(sorted(Counter(x['curriculum_family'] for x in b).items())),'kind_counts':dict(sorted(Counter(x['kind'] for x in b).items()))})
    if len(schedule)!=768 or len({x['case_id'] for x in schedule})!=768:raise RuntimeError('selected schedule geometry drift')
    meta={'selected_case_count':768,'selected_case_order_sha256':hashlib.sha256('\n'.join(x['case_id'] for x in schedule).encode()).hexdigest(),'selected_state_counts':dict(sorted(Counter(x['semantic_root'] for x in schedule).items())),'selected_family_counts':dict(sorted(Counter(x['curriculum_family'] for x in schedule).items())),'selected_kind_counts':dict(sorted(Counter(x['kind'] for x in schedule).items())),'all_minimal_contrast_cases_retained':Counter(x['curriculum_family'] for x in schedule)['minimal_contrast']==512,'checkpoint_blocks':summaries}
    return schedule,meta

def parent_spec(cfg:Mapping[str,Any],parent:Mapping[str,Any])->dict[str,Any]:return sealed.frozen_role_spec(cfg,parent)
