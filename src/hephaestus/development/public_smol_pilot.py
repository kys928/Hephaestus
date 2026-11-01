"""Pinned pretrained Smol public pilot, three real CPU LoRA jobs and receipts.

This material is permanently development-only. It cannot confer a final custody
grant, scientific acceptance, protected-data access or checkpoint promotion.
"""
from collections import Counter
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.schemas.public_pilot import PublicPilotPlan, PublicGenerationContract
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingControlRequest
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.schemas.checkpoint_identity import CheckpointIdentity
from hephaestus.data.chunking import chunk_records
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.training.cpu_chat_json_recipe import chat_prompt
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService, validate_checkpoint_manifest
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.durability import atomic_write_bytes
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.evaluation.prospective_pipeline import digest,file_digest,verify_model_binding
from hephaestus.evaluation.public_pilot import (
    FAMILIES,load_public_model,sample_public_pilot,public_surfaces,generate_public_sample,
    summarize_public_samples,paired_public_summary,verify_public_base)
from .persistence_endurance import _source_inventory


def _publish(repository,artifacts,kind,payload):
    ref=artifacts.put_bytes(canonical_bytes(payload)).artifact_ref
    repository.append('public_pilot_evidence',{'kind':kind,'artifact_ref':ref},operation_id='public-pilot-'+kind)
    return ref


def prepare_public_proposal(root,*,seed,model_path,tokenizer,rows,parameter_count):
    """Exactly the original 96-example interleaved chat/EOS epoch, without truncation."""
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    if (config.model_type,config.hidden_size,config.num_hidden_layers,config.vocab_size,
            config.max_position_embeddings,parameter_count) != ('llama',576,30,49152,8192,134515008):
        raise PermissionError('public pilot proposal must use the original Smol architecture and parameter count')
    root.mkdir(parents=True,exist_ok=False)
    by_family={f:[r for r in rows if r['family']==f] for f in FAMILIES}
    if any(len(v)!=24 for v in by_family.values()):
        raise ValueError('public pilot requires the complete original 96-example train split')
    ordered=[by_family[f][i] for i in range(24) for f in FAMILIES]
    template=chat_prompt(tokenizer,'{prompt}')+'{target}'
    records=chunk_records(tuple({'prompt':r['prompt'],'target':r['target']+tokenizer.eos_token} for r in ordered),
        chunk_size_tokens=8192,min_tokens=1,prompt_target_template=template).records
    if any(len(tokenizer.encode(r['text'],add_special_tokens=False))>8192 for r in records):
        raise ValueError('public training must not truncate any example')
    data=root/'trainable.jsonl'; atomic_write_bytes(data,b''.join(canonical_bytes(r)+b'\n' for r in records),overwrite=False)
    run_id=f'public-pilot-seed-{seed}'
    contract=root/'contract.json'
    atomic_write_bytes(contract,canonical_bytes(TrainableDataContract('public-pilot-'+str(seed),run_id,
        'original-public-96-chat-json-epoch',str(data),'trainable-data.v1',1).to_dict()),overwrite=False)
    evidence=root/'processing.json'
    atomic_write_bytes(evidence,canonical_bytes({'processed_dataset_ref':str(data),'processed_content_hash':file_digest(data),
        'wrapper':{'kind':'explicit_prompt_target_template','template':template},
        'prompt_target_boundary':{'status':'explicit','schema_version':'prompt-target-structured.v1'},
        'tokenizer_compatibility':{'status':'checked','compatible':True,'tokenizer_ref':str(model_path),
            'checker_id':'verified-pinned-smol-chat-eos-no-truncation.v1'}}),overwrite=False)
    identity=directory_content_identity(model_path)
    special={key:getattr(tokenizer,key) for key in ('pad_token_id','eos_token_id','unk_token_id','bos_token_id')}
    # Smol has no unknown token; explicit absent IDs are not integer IDs.
    special={key:value for key,value in special.items() if value is not None}
    return ExperimentProposal(experiment_id=run_id,run_id=run_id,lineage_id='public-pilot-'+str(seed),
        stage_name='smoke_test',diagnosis_report_id='public-nonconfirmatory-pilot',intervention_id=run_id,
        primary_variable='training_seed',baseline_ref=identity,status='ready',training_constraints={
            'backend_id':'transformers_causal_lm','model_id':str(model_path),'model_revision':identity,
            'tokenizer_id':str(model_path),'tokenizer_ref':str(model_path),'tokenizer_revision':identity,
            'architecture_family':config.model_type,'training_mode':'lora_finetune',
            'peft_config':{'rank':8,'alpha':16.,'dropout':0.,'target_modules':['q_proj','v_proj']},
            'trainable_data_contract_ref':str(contract),'trainable_data_contract_hash':file_digest(contract),
            'processed_dataset_ref':str(data),'processed_dataset_hash':file_digest(data),
            'processing_evidence_ref':str(evidence),'processing_evidence_hash':file_digest(evidence),
            'optimizer':'adamw','scheduler':'constant','parameter_count':parameter_count,
            'vocabulary_size':config.vocab_size,'hidden_size':config.hidden_size,'special_token_ids':special,
            'context_length':config.max_position_embeddings,'device':'cpu','dtype':'float32','seed':seed,
            'learning_rate':.0005,'batch_size':1,'gradient_accumulation_steps':1,'max_steps':96,
            'checkpoint_every_steps':96,'logging_every_steps':1,'max_total_tokens':100000,
            'local_files_only':True,'shuffle':False})


def _compare_replay(original,replay):
    import torch
    from safetensors.torch import load_file
    a,b=(load_file(str(Path(p)/'model'/'adapter_model.safetensors')) for p in (original,replay))
    if set(a)!=set(b) or any(not torch.equal(a[k],b[k]) for k in a):
        raise ValueError('same-seed public adapter training did not reproduce exactly')
    states=[torch.load(Path(p)/'training_state.pt',map_location='cpu',weights_only=True) for p in (original,replay)]
    def equal(x,y):
        if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
        if isinstance(x,dict):return isinstance(y,dict) and set(x)==set(y) and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return type(x) is type(y) and len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y,strict=True))
        return x==y
    fields=('step','examples_processed','tokens_processed','batches_processed','last_training_loss',
            'last_gradient_norm','config_fingerprint','optimizer_state','scheduler_state','python_rng_state','torch_rng_state')
    if any(not equal(states[0][key],states[1][key]) for key in fields):
        raise ValueError('same-seed optimizer, scheduler, RNG or dose differs')
    return {'all_adapter_tensors_equal':True,'adapter_tensors':len(a),
            'optimizer_scheduler_rng_and_dose_equal':True,'updates_reexecuted':96}


def run_public_pilot(root,*,original_model,historical_adapter,source_commit):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    root=Path(root).absolute(); root.mkdir(parents=True,exist_ok=False)
    plan=PublicPilotPlan(); generation=PublicGenerationContract(); started=time.monotonic()
    inventory=_source_inventory()
    model_path=root/'public-base'; shutil.copytree(original_model,model_path,symlinks=False)
    historical_path=root/'historical-public-cpu-adapter'; shutil.copytree(historical_adapter,historical_path,symlinks=False)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    verify_public_base(model_path,tokenizer)
    with torch.device('meta'):
        skeleton=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False))
    parameter_count=sum(p.numel() for p in skeleton.parameters()); del skeleton
    original,_=generate_original_splits(GeneratedDataSpec(20261028))
    train,development=original['train'],original['development']
    if (digest({'schema':'original-task-examples.v1','split':'development','examples':development})
            != 'sha256:4c9d54866719545f9a54466cd6da7b9aed4b79f935160d61e42e0639675c3f1f'
            or 'sha256:'+hashlib.sha256(b''.join(canonical_bytes(r)+b'\n' for r in train)).hexdigest()
            != 'sha256:602d8fb5a59d2db02cd6a9239a98096ef55c0d906ed7307c1e15f275ec2fb68d'):
        raise PermissionError('public original train/development reconstruction differs')
    excluded={r['sample_id'] for split in original.values() for r in split}
    pilot=sample_public_pilot(plan,excluded); surfaces=public_surfaces(pilot,plan)
    contract={'schema':'public-smol-pilot-contract.v1','plan':plan.to_dict(),'generation':generation.to_dict(),
        'source_commit':source_commit,'source_files':inventory,'calibration_caps':[16,64,128],
        'calibration_rows':'original public development only','model_parameter_count':parameter_count,
        'base_identity':directory_content_identity(model_path),'historical_adapter_identity':directory_content_identity(historical_path),
        'frameworks':{name:version(name) for name in ('torch','transformers','peft','tokenizers','safetensors')},
        'exclusions':sorted(excluded),'pilot_canonical_ids':[r['sample_id'] for r in pilot],
        'source_hashes':{'train':digest(train),'development':digest(development),'pilot':digest(pilot),'surfaces':digest(surfaces)},
        'protected_final_material_exposed':False,'independent_scientific_acceptance':False}
    for name,payload in [('pilot-contract',contract),('original-public-train',{'rows':train}),
                         ('original-public-development',{'rows':development}),('public-pilot',{'rows':pilot,'surfaces':surfaces})]:
        atomic_write_bytes(root/(name+'.json'),canonical_bytes(payload),overwrite=False)
    repository=SQLiteStateRepository(root/'state'/'production.sqlite3',campaign_witness=True)
    artifacts=FileSystemArtifactStore(root/'artifacts')
    _publish(repository,artifacts,'contract',contract)
    bindings=[]; handles=[]
    service=TransformersTrainingLifecycleService(root/'training-runs',maximum_allowed_steps=96)
    for seed in plan.training_seeds:
        with execution_owner(repository,'public-pilot-training',str(seed)):
            proposal=prepare_public_proposal(root/f'inputs-{seed}',seed=seed,model_path=model_path,
                tokenizer=tokenizer,rows=train,parameter_count=parameter_count)
            _publish(repository,artifacts,f'seed-{seed}-prepared',proposal.to_dict())
            result=service.launch(proposal); deadline=time.monotonic()+1800
            while result.status in {'preparing','queued','running','resuming'} and time.monotonic()<deadline:
                time.sleep(.2); result=service.status(proposal.run_id)
            if result.status!='completed':
                service.control(TrainingControlRequest('cancel-'+proposal.run_id,proposal.run_id,'cancel','public-pilot','CPU deadline/failure'))
                _publish(repository,artifacts,f'seed-{seed}-failed',result.to_dict())
                raise RuntimeError('public pilot training did not complete; retain this attempt')
            checkpoint=Path(result.checkpoint_refs[-1]); handles.append(result.to_dict())
            handle_ref=_publish(repository,artifacts,f'seed-{seed}-completed',result.to_dict())
            valid,identity=validate_checkpoint_manifest(checkpoint)
            if not valid:raise ValueError('public pilot checkpoint failed producer manifest validation')
            CheckpointIdentityStore(repository).retain(CheckpointIdentity('public-pilot-'+str(seed),str(checkpoint),identity,handle_ref))
            job=checkpoint.parent/'prepared_job.json'; base_identity=directory_content_identity(model_path)
            binding=ProspectiveModelBinding(f'seed-{seed}',str(model_path),base_identity,str(model_path),base_identity,
                seed,str(checkpoint/'model'),directory_content_identity(checkpoint/'model'),str(checkpoint),identity,str(job),file_digest(job))
            # This is intentionally a public-pilot purpose, never confirmatory.
            verify_model_binding(binding,plan)
            state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
            if state['step']!=96 or state['examples_processed']!=96:raise ValueError('pilot training dose differs')
            bindings.append(binding)
            print(json.dumps({'phase':'training-complete','seed':seed,'updates':96}),flush=True)
    first=next(b for b in bindings if b.seed==plan.same_seed_replay)
    replay=root/'same-seed-replay'; replay.mkdir()
    job=json.loads(Path(first.job_path).read_text()); job['artifact_root']=str(replay)
    atomic_write_bytes(replay/'job.json',canonical_bytes(job),overwrite=False)
    with (replay/'stdout.log').open('wb') as out,(replay/'stderr.log').open('wb') as error:
        completed=subprocess.run([sys.executable,'-m','hephaestus.training.hf_worker','--job',str(replay/'job.json')],
                                 stdout=out,stderr=error,timeout=1800,check=False)
    if completed.returncode!=0:raise RuntimeError('same-seed replay failed; retain all bytes')
    reproducibility=_compare_replay(first.checkpoint_path,replay/'checkpoint_step_96')
    replay_valid,replay_identity=validate_checkpoint_manifest(replay/'checkpoint_step_96')
    if not replay_valid:raise ValueError('replay checkpoint incomplete')
    replay_ref=_publish(repository,artifacts,'same-seed-reproducibility',reproducibility)
    CheckpointIdentityStore(repository).retain(CheckpointIdentity('public-pilot-replay',str(replay/'checkpoint_step_96'),replay_identity,replay_ref))
    outputs={}; development_outputs={}; calibration={}; model_receipts={}
    models=[('baseline',None,False),('historical-public-cpu-adapter',historical_path,True)]
    models += [(b.label,Path(b.adapter_path),False) for b in bindings]
    for label,adapter,historical in models:
        with execution_owner(repository,'public-pilot-generation',label):
            model,loaded_tokenizer,info=load_public_model(model_path,adapter_path=adapter,historical=historical)
            dev_by_cap={}
            for cap in ([16,64,128] if label in {'baseline','historical-public-cpu-adapter'} else [128]):
                cap_contract=PublicGenerationContract(contract_id=f'smol-public-json-cap{cap}.v1',max_new_tokens=cap)
                samples=[generate_public_sample(model,loaded_tokenizer,row,cap_contract) for row in development]
                dev_by_cap[str(cap)]={'contract':cap_contract.to_dict(),'samples':samples,'summary':summarize_public_samples(samples)}
            development_outputs[label]=dev_by_cap['128']
            if len(dev_by_cap)>1:calibration[label]=dev_by_cap
            samples=[generate_public_sample(model,loaded_tokenizer,row,generation) for row in surfaces]
            output={'model_label':label,'samples':samples,'summary':summarize_public_samples(samples),
                'by_family':{f:summarize_public_samples([s for s in samples if s['family']==f and s['template_id']=='original-canonical.v1']) for f in FAMILIES},
                'by_difficulty':{group:summarize_public_samples([s for s in samples
                    if s['task_characteristics']['difficulty_group']==group and s['template_id']=='original-canonical.v1'])
                    for group in sorted({s['task_characteristics']['difficulty_group'] for s in samples})},
                'by_template':{t:summarize_public_samples([s for s in samples if s['template_id']==t])
                               for t in ('original-canonical.v1','task-wrapper-sensitivity.v1')},
                'loader_info':info,'contract':generation.to_dict(),'confirmatory':False}
            ref=_publish(repository,artifacts,'generation-'+label,output)
            model_receipts[label]=ref; outputs[label]=output
            print(json.dumps({'phase':'generation-complete','label':label,'summary':output['by_template']['original-canonical.v1']}),flush=True)
            del model; gc.collect()
    attempts=[{'seed':b.seed,'development_terminated_exact':development_outputs[b.label]['summary']['terminated_exact'],
               'development_valid_json':development_outputs[b.label]['summary']['valid_json']} for b in bindings]
    selected=max(attempts,key=lambda row:(row['development_terminated_exact'],row['development_valid_json'],-row['seed']))['seed']
    canonical=lambda label:[s for s in outputs[label]['samples'] if s['template_id']=='original-canonical.v1']
    analysis={'schema':'public-smol-pilot-analysis.v1','purpose':plan.purpose,'contract':contract,
        'training_updates':288,'replay_updates':96,'training_reproducibility':reproducibility,
        'development_seed_attempts':attempts,'development_selected_seed':selected,
        'generation_refs':model_receipts,'calibration':calibration,'development':development_outputs,
        'canonical_pilot':{label:outputs[label]['by_template']['original-canonical.v1'] for label,_,_ in models},
        'by_family':{label:outputs[label]['by_family'] for label,_,_ in models},
        'by_difficulty':{label:outputs[label]['by_difficulty'] for label,_,_ in models},
        'template_sensitivity':{label:outputs[label]['by_template'] for label,_,_ in models},
        'paired_canonical':{label:paired_public_summary(canonical('baseline'),canonical(label)) for label,_,_ in models if label!='baseline'},
        'elapsed_seconds':time.monotonic()-started,'confirmatory_improvement_established':False,
        'external_certification':False,'protected_final_holdout_exposed':False}
    if (_source_inventory()!=inventory or directory_content_identity(model_path)!=contract['base_identity']
            or directory_content_identity(historical_path)!=contract['historical_adapter_identity']):
        raise ValueError('public pilot source or fixed pretrained bytes changed during execution')
    analysis_ref=_publish(repository,artifacts,'analysis',analysis)
    atomic_write_bytes(root/'analysis.json',canonical_bytes({'analysis_ref':analysis_ref,'analysis':analysis}),overwrite=False)
    atomic_write_bytes(root/'bindings.json',canonical_bytes({'bindings':[b.to_dict() for b in bindings]}),overwrite=False)
    return {'analysis_ref':analysis_ref,'root':str(root),'paired_canonical':analysis['paired_canonical'],
            'canonical_pilot':analysis['canonical_pilot'],'training_reproducibility':reproducibility,
            'protected_final_holdout_exposed':False}


def cold_public_pilot(root,*,expected_analysis_ref,reexecute=True):
    """Read all raw token receipts; rerun a predeclared paired subset on each model."""
    import torch
    from transformers import AutoTokenizer
    from hephaestus.evaluation.exact_json import score_prediction
    from hephaestus.training.cpu_chat_json_recipe import chat_prompt
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    root=Path(root).absolute()
    repository=SQLiteStateRepository(root/'state'/'production.sqlite3',read_only=True)
    repository.assert_integrity()
    artifacts=FileSystemArtifactStore(root/'artifacts')
    analysis=read_verified_json(artifacts,expected_analysis_ref)
    contract=analysis['contract']; plan=PublicPilotPlan.from_dict(contract['plan'])
    generation=PublicGenerationContract.from_dict(contract['generation'])
    if _source_inventory()!=contract['source_files']:
        raise ValueError('cold public pilot requires the exact frozen source installation')
    published=json.loads((root/'analysis.json').read_text())
    if published!={'analysis_ref':expected_analysis_ref,'analysis':analysis}:
        raise ValueError('cold public analysis differs from separately retained publication')
    train=json.loads((root/'original-public-train.json').read_text())['rows']
    dev=json.loads((root/'original-public-development.json').read_text())['rows']
    pilot_data=json.loads((root/'public-pilot.json').read_text())
    rows,surfaces=pilot_data['rows'],pilot_data['surfaces']
    if ({'train':digest(train),'development':digest(dev),'pilot':digest(rows),'surfaces':digest(surfaces)}
            !=contract['source_hashes'] or rows!=sample_public_pilot(plan,set(contract['exclusions']))
            or surfaces!=public_surfaces(rows,plan)):
        raise ValueError('cold public sampling/partition or frozen input bytes differ')
    model_path=root/'public-base'; historical=root/'historical-public-cpu-adapter'
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    verify_public_base(model_path,tokenizer)
    if (directory_content_identity(model_path)!=contract['base_identity']
            or directory_content_identity(historical)!=contract['historical_adapter_identity']):
        raise ValueError('cold public model bytes differ')
    bindings=[ProspectiveModelBinding.from_dict(b) for b in json.loads((root/'bindings.json').read_text())['bindings']]
    if {b.seed for b in bindings}!=set(plan.training_seeds):raise ValueError('cold pilot seed roster incomplete')
    for binding in bindings:verify_model_binding(binding,plan)
    by_surface={row['surface_id']:row for row in surfaces}
    selected_ids={min(r['sample_id'] for r in rows if r['family']==f) for f in FAMILIES}
    outputs={}; verified_tokens=0; regenerated=0
    def verify_samples(samples,items,budget):
        nonlocal verified_tokens
        if len(samples)!=len(items):raise ValueError('cold public generation incomplete')
        for sample,row in zip(samples,items,strict=True):
            tokens=sample['generated_tokens']; eos=bool(tokens and tokens[-1]==2)
            if (sample['sample_id']!=row['sample_id'] or not 1<=len(tokens)<=budget
                    or any(type(t) is not int or not 0<=t<len(tokenizer) for t in tokens) or 2 in tokens[:-1]
                    or sample['eos']!=eos or sample['truncated']!=(len(tokens)==budget and not eos)):
                raise ValueError('cold public token identity or termination receipt differs')
            prompt=chat_prompt(tokenizer,row.get('model_prompt',row['prompt']))
            body=tokens[:-1] if eos else tokens
            answer=tokenizer.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            scored=score_prediction(answer,row['target'])
            if (sample['prompt_tokens']!=tokenizer.encode(prompt,add_special_tokens=False)
                    or sample['prompt_sha256']!='sha256:'+hashlib.sha256(prompt.encode()).hexdigest()
                    or sample['answer_text']!=answer or sample['raw_decode']!=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    or sample['typed_exact']!=scored['exact_correct'] or sample['valid_json']!=scored['valid_json']
                    or sample['terminated_exact']!=bool(eos and scored['exact_correct'])
                    or sample['target_tokens']!=tokenizer.encode(row['target']+tokenizer.eos_token,add_special_tokens=False)):
                raise ValueError('cold public prompt, decoding or strict score differs')
            verified_tokens+=len(tokens)
    for label,ref in analysis['generation_refs'].items():
        output=read_verified_json(artifacts,ref); samples=output['samples']
        verify_samples(samples,surfaces,128)
        if (output['summary']!=summarize_public_samples(samples)
                or output['contract']!=generation.to_dict() or output['confirmatory'] is not False):
            raise ValueError('cold public output summary/contract differs')
        verify_samples(analysis['development'][label]['samples'],dev,128)
        if analysis['development'][label]['summary']!=summarize_public_samples(analysis['development'][label]['samples']):
            raise ValueError('cold development scores differ')
        if label in analysis['calibration']:
            for cap,record in analysis['calibration'][label].items():
                verify_samples(record['samples'],dev,int(cap))
                if record['summary']!=summarize_public_samples(record['samples']):raise ValueError('cold calibration summary differs')
        outputs[label]=output
        if reexecute:
            adapter=historical if label=='historical-public-cpu-adapter' else next((Path(b.adapter_path) for b in bindings if b.label==label),None)
            model,loaded,_=load_public_model(model_path,adapter_path=adapter,historical=label=='historical-public-cpu-adapter')
            for sample in samples:
                if sample['sample_id'] in selected_ids:
                    if generate_public_sample(model,loaded,by_surface[sample['surface_id']],generation)!=sample:
                        raise ValueError('cold fixed-model public generation/NLL reexecution differs')
                    regenerated+=1
            del model; gc.collect()
    canonical=lambda label:[s for s in outputs[label]['samples'] if s['template_id']=='original-canonical.v1']
    paired={label:paired_public_summary(canonical('baseline'),canonical(label)) for label in outputs if label!='baseline'}
    if paired!=analysis['paired_canonical']:raise ValueError('cold public paired analysis differs')
    attempts=[{'seed':b.seed,'development_terminated_exact':analysis['development'][b.label]['summary']['terminated_exact'],
               'development_valid_json':analysis['development'][b.label]['summary']['valid_json']} for b in bindings]
    selected=max(attempts,key=lambda row:(row['development_terminated_exact'],row['development_valid_json'],-row['seed']))['seed']
    if selected!=analysis['development_selected_seed'] or attempts!=analysis['development_seed_attempts']:
        raise ValueError('cold development-only seed selection differs')
    first=next(b for b in bindings if b.seed==plan.same_seed_replay)
    replay=_compare_replay(first.checkpoint_path,root/'same-seed-replay'/'checkpoint_step_96')
    if replay!=analysis['training_reproducibility']:raise ValueError('cold same-seed optimization state differs')
    return {'schema':'public-smol-pilot-cold.v1','analysis_ref':expected_analysis_ref,'verified':True,
        'all_raw_tokens_decoded_and_scored':True,'generated_tokens_verified':verified_tokens,
        'fixed_model_samples_reexecuted':regenerated,'all_retained_training_state_rechecked':True,
        'protected_final_holdout_exposed':False,'external_independent_certification':False}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','cold'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--original-model',type=Path)
    parser.add_argument('--historical-adapter',type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--expected-analysis-ref')
    args=parser.parse_args()
    if args.action=='run':
        if not all((args.original_model,args.historical_adapter,args.source_commit)):parser.error('run requires original source paths and source commit')
        result=run_public_pilot(args.root,original_model=args.original_model,historical_adapter=args.historical_adapter,source_commit=args.source_commit)
    else:
        if not args.expected_analysis_ref:parser.error('cold requires the separately retained analysis reference')
        result=cold_public_pilot(args.root,expected_analysis_ref=args.expected_analysis_ref)
    print(json.dumps(result,indent=2))
