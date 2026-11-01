"""Raw public Smol generation and paired feasibility; no holdout construction."""
import hashlib
import json
from pathlib import Path
import random
import statistics

from hephaestus.schemas.public_pilot import PublicGenerationContract, PublicPilotPlan
from hephaestus.data.generated_factory import _example
from hephaestus.data.generated_validation import validate_original_example, parse_and_solve
from hephaestus.evaluation.exact_json import score_prediction, SCORER_VERSION
from hephaestus.evaluation.prospective_pipeline import digest, file_digest, ORIGINAL_WEIGHTS, ORIGINAL_TOKENIZER
from hephaestus.storage.verified import canonical_bytes
from hephaestus.training.cpu_chat_json_recipe import chat_prompt

ORIGINAL_TEMPLATE = 'sha256:872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788'
HISTORICAL_PUBLIC_ADAPTER = 'sha256:a782d0ca344559659893a64d8bc1e8a90f597cf9cf0bbef06b433c555c1da195'
FAMILIES = ('arithmetic','json','symbolic','reasoning')
PUBLIC_SNAPSHOT_FILES = {
    'README.md':'4f97533ad95b1b2fea15fbc075c01b94578ebdd7c8138888fa43fa3abd530dc4',
    'config.json':'8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a',
    'generation_config.json':'87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013',
    'model.safetensors':ORIGINAL_WEIGHTS[7:], 'tokenizer.json':ORIGINAL_TOKENIZER[7:],
    'special_tokens_map.json':'2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52',
    'tokenizer_config.json':'4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df'}


def sample_public_pilot(plan: PublicPilotPlan, excluded_ids):
    plan.__post_init__()
    if not excluded_ids:
        raise ValueError('original public train/development exposures must be supplied')
    rng, seen, rows = random.Random(plan.sampling_seed), set(excluded_ids), []
    for family in FAMILIES:
        accepted = 0
        for _ in range(100000):
            row = _example(rng,family,plan.difficulty)
            identity = digest(validate_original_example(row))
            # Only development-bucket identities are retained. No final seed,
            # permission or protected batch is available to this function.
            if int(identity[7:],16)%10 != plan.canonical_bucket or identity in seen:
                continue
            rows.append({**row,'sample_id':identity,'purpose':plan.purpose})
            seen.add(identity); accepted += 1
            if accepted == plan.cases_per_family:
                break
        else:
            raise ValueError('public development pilot support exhausted')
    return rows


def public_surfaces(rows, plan):
    surfaces = [{**row,'model_prompt':row['prompt'],'template_id':'original-canonical.v1'} for row in rows]
    for family in FAMILIES:
        selected=sorted((row for row in rows if row['family']==family),key=lambda row:row['sample_id'])[:plan.template_pairs_per_family]
        surfaces += [{**row,'model_prompt':'Task:\n'+row['prompt']+'\nReply with just the requested JSON object.',
                      'template_id':'task-wrapper-sensitivity.v1'} for row in selected]
    for row in surfaces:
        row['surface_id']=digest({'canonical_id':row['sample_id'],'template_id':row['template_id'],'prompt':row['model_prompt']})
    return surfaces


def verify_public_base(path, tokenizer):
    if (any(file_digest(Path(path)/name) != 'sha256:'+expected for name,expected in PUBLIC_SNAPSHOT_FILES.items())
            or 'sha256:'+hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != ORIGINAL_TEMPLATE
            or tokenizer.eos_token_id != 2 or tokenizer.pad_token_id != 2):
        raise PermissionError('public original model, tokenizer, template or special tokens differ')


def load_public_model(path, *, adapter_path=None, historical=False):
    """Validate every adapter tensor before and after loading, without fallback."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    verify_public_base(path,tokenizer)
    model, info=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                                                  dtype=torch.float32,output_loading_info=True)
    if info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise PermissionError('public base loader reported incomplete/incompatible parameters')
    missing=set(info.get('missing_keys',[]))
    if missing and not (missing=={'lm_head.weight'} and model.config.tie_word_embeddings
            and model.get_output_embeddings().weight.data_ptr()==model.get_input_embeddings().weight.data_ptr()):
        raise PermissionError('public base is missing required parameters')
    if adapter_path is not None:
        from peft import get_peft_model, LoraConfig
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        adapter_path=Path(adapter_path)
        if historical and file_digest(adapter_path/'adapter_model.safetensors') != HISTORICAL_PUBLIC_ADAPTER:
            raise PermissionError('historical public CPU adapter bytes differ')
        cfg=json.loads((adapter_path/'adapter_config.json').read_text())
        if (cfg.get('r')!=8 or cfg.get('lora_alpha')!=16 or cfg.get('lora_dropout')!=0
                or set(cfg.get('target_modules',[]))!={'q_proj','v_proj'} or cfg.get('bias')!='none'
                or cfg.get('peft_type')!='LORA' or cfg.get('task_type')!='CAUSAL_LM'
                or any(cfg.get(k) for k in ('use_dora','use_rslora','modules_to_save','rank_pattern','alpha_pattern'))):
            raise PermissionError('public adapter recipe differs from intended rank8/alpha16 q/v recipe')
        base=str(cfg.get('base_model_name_or_path','')).rstrip('/')
        suffix='/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac'
        if base != str(Path(path)).rstrip('/') and not (historical and base.endswith(suffix)):
            raise PermissionError('adapter names a different base model')
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
                    target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
        expected=get_peft_model_state_dict(model)
        actual=load_file(str(adapter_path/'adapter_model.safetensors'),device='cpu')
        if (set(actual)!=set(expected) or any(actual[k].shape!=expected[k].shape
                or not torch.isfinite(actual[k]).all() for k in actual)):
            raise PermissionError('adapter tensor key set, shapes or values are incompatible')
        set_peft_model_state_dict(model,actual)
        loaded=get_peft_model_state_dict(model)
        if any(not torch.equal(loaded[k].detach().cpu(),actual[k].to(loaded[k].dtype)) for k in actual):
            raise PermissionError('adapter tensors did not load exactly')
    model.to('cpu').eval()
    def json_safe(value):
        if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
        if isinstance(value,set):return [json_safe(item) for item in sorted(value,key=str)]
        if isinstance(value,(tuple,list)):return [json_safe(item) for item in value]
        return value
    return model,tokenizer,json_safe(info)


def task_characteristics(row):
    key,answer=parse_and_solve(row['prompt'])
    if key[0] in {'arithmetic','json'}:
        values=key[2] if key[0]=='arithmetic' else key[1]
        largest=max(values)
        band='max-input-0to19' if largest<20 else 'max-input-20to49' if largest<50 else 'max-input-50to99'
        return {'difficulty_group':key[0]+'/'+band,'maximum_input':largest,
                **({'operation':key[1]} if key[0]=='arithmetic' else {'list_length':len(values)})}
    if key[0]=='symbolic':
        length=len(key[1])
        return {'difficulty_group':'symbolic/'+('length3to5' if length<=5 else 'length6to8'),'string_length':length}
    return {'difficulty_group':'reasoning/distance'+str(answer['distance']),
            'shortest_path_length':answer['distance'],'edges':len(key[1])}


def generate_public_sample(model,tokenizer,row,contract: PublicGenerationContract):
    import torch
    from transformers import GenerationConfig
    contract.__post_init__()
    prefix=chat_prompt(tokenizer,row.get('model_prompt',row['prompt']))
    ids=tokenizer.encode(prefix,add_special_tokens=False)
    target=tokenizer.encode(row['target']+tokenizer.eos_token,add_special_tokens=False)
    if len(ids)+max(contract.max_new_tokens,len(target))>model.config.max_position_embeddings:
        raise ValueError('public pilot forbids prompt or target truncation')
    tensor=torch.tensor([ids],dtype=torch.long)
    # A new config prevents hidden defaults from a model generation_config.json.
    config=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=contract.max_new_tokens,
        min_new_tokens=0,eos_token_id=2,pad_token_id=2,bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=None,forced_bos_token_id=None,repetition_penalty=1.,use_cache=True)
    with torch.inference_mode():
        generated=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=config)
        sequence=torch.tensor([ids+target],dtype=torch.long)
        labels=torch.tensor([[-100]*len(ids)+target],dtype=torch.long)
        nll=float(model(input_ids=sequence,attention_mask=torch.ones_like(sequence),labels=labels).loss)
    tokens=generated[0,len(ids):].tolist()
    eos=bool(tokens and tokens[-1]==2)
    if any(type(token) is not int or not 0<=token<len(tokenizer) for token in tokens) or 2 in tokens[:-1]:
        raise ValueError('generation token receipt violates vocabulary/first-EOS stopping')
    body=tokens[:-1] if eos else tokens
    answer=tokenizer.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    scored=score_prediction(answer,row['target'])
    return {'sample_id':row['sample_id'],'surface_id':row.get('surface_id',row['sample_id']),
        'family':row['family'],'template_id':row.get('template_id','original-canonical.v1'),
        'task_characteristics':task_characteristics(row),
        'prompt_tokens':ids,'prompt_sha256':'sha256:'+hashlib.sha256(prefix.encode()).hexdigest(),
        'generated_tokens':tokens,'target_tokens':target,'eos_token_id':2,'eos':eos,
        'truncated':len(tokens)==contract.max_new_tokens and not eos,
        'raw_decode':tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        'answer_text':answer,'typed_exact':scored['exact_correct'],'valid_json':scored['valid_json'],
        'terminated_exact':bool(scored['exact_correct'] and eos),'target_nll_including_eos':nll}


def summarize_public_samples(samples):
    lengths=sorted(len(s['generated_tokens']) for s in samples)
    if not lengths:
        raise ValueError('cannot summarize an empty public pilot')
    return {'n':len(samples),**{key:sum(bool(s[key]) for s in samples)
            for key in ('typed_exact','terminated_exact','valid_json','eos','truncated')},
            'length_min':min(lengths),'length_median':statistics.median(lengths),
            'length_p90':lengths[min(len(lengths)-1,int(.9*len(lengths)))], 'length_max':max(lengths),
            'mean_target_nll_including_eos':statistics.mean(s['target_nll_including_eos'] for s in samples)}


def paired_public_summary(baseline,candidate):
    if [s['surface_id'] for s in baseline] != [s['surface_id'] for s in candidate]:
        raise ValueError('public paired surface identities differ')
    before=[bool(s['terminated_exact']) for s in baseline]
    after=[bool(s['terminated_exact']) for s in candidate]
    return {'n':len(before),'wins':sum(c and not b for b,c in zip(before,after,strict=True)),
            'losses':sum(b and not c for b,c in zip(before,after,strict=True)),
            'baseline_correct':sum(before),'candidate_correct':sum(after),
            'gain':(sum(after)-sum(before))/len(before),'confirmatory_evidence':False}
