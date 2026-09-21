#!/usr/bin/env python3
"""Phase I independent role-mastery trainer for one frozen Hephaestus role."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,os,random,shutil,tarfile,time,traceback
from pathlib import Path
from typing import Any

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in os.sys.path: os.sys.path.insert(0,str(SCRIPTS))
import run_cognitive_topology_v1 as topo
import run_diagnosis_foundation_bakeoff_v1 as modelio
import launch_first_bounded_scientific_training as storage

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/experiments/hephaestus_role_mastery_v1.json"

def req(name:str)->str:
    v=(os.environ.get(name) or "").strip()
    if not v: raise RuntimeError("missing required environment variable: "+name)
    return v

def put_json(client:Any,key:str,obj:object)->None:
    raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    client.put_object(Bucket=storage.VOLUME_ID,Key=key,Body=raw)
    if storage.read_key(client,key)!=raw: raise RuntimeError("S3 readback mismatch: "+key)

def put_jsonl(client:Any,key:str,rows:list[dict[str,Any]])->None:
    raw=("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in rows)).encode()
    client.put_object(Bucket=storage.VOLUME_ID,Key=key,Body=raw)
    if storage.read_key(client,key)!=raw: raise RuntimeError("S3 readback mismatch: "+key)

def load_pack(cfg:dict[str,Any])->tuple[Any,dict[str,Any]]:
    p=ROOT/cfg["pack"]["builder_path"]
    s=importlib.util.spec_from_file_location("role_mastery_pack",p)
    if s is None or s.loader is None: raise RuntimeError("cannot import role mastery pack")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    pack=m.build_pack();m.validate(pack)
    observed=m.canonical_sha256(pack)
    if observed!=cfg["pack"]["canonical_sha256"]: raise RuntimeError(f"pack hash drift {observed}")
    return m,pack

def role_prompt(pack:dict[str,Any],case:dict[str,Any])->str:
    role=case["role"];v=pack["contract_vocabulary"][role]
    evidence="\n".join(f"- {x['ref']}: {x['fact']}" for x in case["evidence"])
    keys=", ".join(pack["response_schema"]["required_exact_keys"])
    return f"""You are the Hephaestus {role.upper()} specialist.

ROLE CONTRACT:
{pack['role_rules'][role]}

SITUATION:
{case['situation']}

EVIDENCE:
{evidence}

Return exactly one JSON object and nothing else. No markdown.
The object must contain exactly these keys: {keys}.
Use only the role vocabulary:
- decision: {', '.join(v['decision'])}
- action: {', '.join(v['action'])}
- primary_variable: {', '.join(v['primary_variable'])}
- confidence: number in [0,1]
- evidence_refs: cite only material evidence from this case
- uncertainties: short JSON array
- rationale: one concise evidence-grounded sentence

Do not perform another role's job. Do not invent evidence. Do not reveal hidden reasoning."""

def template_ids(tokenizer:Any,cand:dict[str,Any],messages:list[dict[str,str]],generation_prompt:bool)->list[int]:
    kwargs=dict(cand.get("chat_template_kwargs") or {})
    if cand["load_kind"]=="mistral3":
        cont=bool(messages and messages[-1].get("role")=="assistant" and not generation_prompt)
        enc=tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True,continue_final_message=cont,**kwargs)
    else:
        enc=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation_prompt,return_tensors="pt",return_dict=True,**kwargs)
    return [int(x) for x in enc["input_ids"][0].tolist()]

def training_example(tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],builder:Any,case:dict[str,Any],maxlen:int)->dict[str,Any]:
    import torch
    user=role_prompt(pack,case)
    answer=json.dumps(builder.target(case),ensure_ascii=False,separators=(",",":"))
    p=template_ids(tokenizer,cand,[{"role":"user","content":user}],True)
    f=template_ids(tokenizer,cand,[{"role":"user","content":user},{"role":"assistant","content":answer}],False)
    lcp=0
    for a,b in zip(p,f):
        if a!=b: break
        lcp+=1
    if lcp<8: raise RuntimeError(f"chat template prefix mismatch {cand['candidate_id']} {case['case_id']} lcp={lcp}")
    if len(f)>maxlen: raise RuntimeError(f"training example {case['case_id']} length {len(f)}>{maxlen}")
    pad=getattr(tokenizer,"pad_token_id",None)
    if pad is None: pad=getattr(tokenizer,"eos_token_id",None)
    if isinstance(pad,(list,tuple)): pad=pad[0] if pad else None
    if pad is None: raise RuntimeError("no pad/eos token id")
    ids=f+[int(pad)]*(maxlen-len(f));mask=[1]*len(f)+[0]*(maxlen-len(f))
    labels=[-100]*min(lcp,len(f))+f[min(lcp,len(f)):]
    labels+= [-100]*(maxlen-len(labels))
    return {"input_ids":torch.tensor([ids]),"attention_mask":torch.tensor([mask]),"labels":torch.tensor([labels]),"nonpad_tokens":len(f),"supervised_tokens":sum(x!=-100 for x in labels)}

def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    skills={}
    for skill in sorted({r["skill"] for r in rows}):
        xs=[r for r in rows if r["skill"]==skill]
        skills[skill]={
          "sample_count":len(xs),
          "quality_100":mean([x["score"]["quality_100"] for x in xs]),
          "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in xs]),
          "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in xs]),
          "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in xs]),
          "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in xs]),
          "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in xs]),
        }
    return {
      "sample_count":len(rows),
      "quality_100":mean([x["score"]["quality_100"] for x in rows]),
      "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in rows]),
      "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in rows]),
      "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in rows]),
      "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in rows]),
      "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in rows]),
      "mean_latency_seconds":mean([x["generation"]["total_latency_seconds"] for x in rows]),
      "mean_generated_tokens":mean([x["generation"]["generated_tokens"] for x in rows]),
      "peak_eval_vram_bytes":max([x["generation"]["peak_vram_bytes"] for x in rows],default=0),
      "skills":skills,
    }

def evaluate(model:Any,tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cases:list[dict[str,Any]],seed:int,max_new:int,deadline:float,client:Any,key:str,heartbeat,phase:str)->dict[str,Any]:
    rows=[]
    model.eval()
    for i,case in enumerate(cases,1):
        if time.monotonic()>=deadline: raise TimeoutError("role mastery hard wall during evaluation")
        gen=modelio.generate(model,tokenizer,cand,role_prompt(pack,case),seed,max_new,deadline)
        raw=gen.pop("raw_output")
        normalized,extract=modelio.extract_complete_json(raw)
        score=topo.score_response(pack,case,normalized)
        rows.append({"case_id":case["case_id"],"root_case_id":case["root_case_id"],"skill":case["skill"],"phase":phase,"raw_output":raw,"output":normalized,"extraction":extract,"generation":gen,"score":score})
        if i==1 or i%25==0 or i==len(cases):
            heartbeat(f"{phase}_evaluation",completed=i,total=len(cases),quality_so_far=summarize(rows)["quality_100"])
    put_jsonl(client,key,rows)
    return summarize(rows)

def set_training_mode(model:Any,enabled:bool)->None:
    if enabled:
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
        if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
        if hasattr(model.config,"use_cache"): model.config.use_cache=False
        model.train()
    else:
        if hasattr(model,"gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
        if hasattr(model.config,"use_cache"): model.config.use_cache=True
        model.eval()

def save_adapter(model:Any,path:Path)->None:
    path.mkdir(parents=True,exist_ok=True);model.save_pretrained(path,safe_serialization=True)

def archive_adapter(path:Path,out:Path,client:Any,key:str)->dict[str,Any]:
    with tarfile.open(out,"w:gz") as tf: tf.add(path,arcname="adapter")
    h=hashlib.sha256()
    with out.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):h.update(chunk)
    client.upload_file(str(out),storage.VOLUME_ID,key)
    head=client.head_object(Bucket=storage.VOLUME_ID,Key=key)
    return {"s3_key":key,"sha256":h.hexdigest(),"bytes":int(head["ContentLength"])}

def certification(summary:dict[str,Any],thresholds:dict[str,Any])->dict[str,Any]:
    min_skill=min((v["quality_100"] for v in summary["skills"].values()),default=0.0)
    gates={
      "quality_100":summary["quality_100"]>=float(thresholds["quality_100"]),
      "schema":summary["schema_compliance"]>=float(thresholds["schema"]),
      "grounding":summary["evidence_grounding"]>=float(thresholds["grounding"]),
      "hallucination":summary["hallucination_rate"]<=float(thresholds["hallucination_max"]),
      "min_skill_quality":min_skill>=float(thresholds["min_skill_quality"]),
    }
    return {"passed":all(gates.values()),"gates":gates,"thresholds":thresholds,"observed_min_skill_quality":min_skill}

def main()->int:
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model
    ap=argparse.ArgumentParser();ap.add_argument("--role",default=os.environ.get("HEPHAESTUS_ROLE"));args=ap.parse_args()
    role=str(args.role or "").strip()
    cfg=json.loads(CFG_PATH.read_text())
    if role not in cfg["candidates"]: raise RuntimeError(f"unknown role {role}")
    builder,pack=load_pack(cfg);cand=dict(cfg["candidates"][role])
    run_id=req("HEPHAESTUS_ROLE_MASTERY_RUN_ID");repo_sha=req("HEPHAESTUS_REPO_SHA")
    client=storage.s3_client();prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}/roles/{role}"
    deadline=time.monotonic()+int(cfg["execution"]["hard_wall_seconds"])
    root=Path("/opt/hephaestus-role-mastery")/run_id/role;root.mkdir(parents=True,exist_ok=True)
    result={"result_version":"hephaestus-role-mastery.v1","status":"running","role":role,"run_id":run_id,"repo_sha":repo_sha,"pack_sha256":cfg["pack"]["canonical_sha256"],"model_id":cand["model_id"],"revision":cand["revision"],"production_promotion_performed":False}
    def heartbeat(stage:str,**extra):
        p={"run_id":run_id,"role":role,"stage":stage,"timestamp_unix":time.time(),"remaining_wall_seconds":max(0,deadline-time.monotonic()),**extra}
        put_json(client,f"{prefix}/progress.json",p);print("ROLE_MASTERY_HEARTBEAT_JSON "+json.dumps(p,sort_keys=True),flush=True)
    try:
        heartbeat("materializing_model")
        snap,adapter,manifest=modelio.materialize(cand,root/"model")
        put_json(client,f"{prefix}/model_manifest.json",manifest)
        model,tokenizer,runtime=modelio.load(cand,snap,adapter,cfg);put_json(client,f"{prefix}/runtime.json",runtime)
        maxlen=int(cfg["training"]["max_sequence_length"])
        train_cases=pack["splits"][role]["train"];dev_cases=pack["splits"][role]["dev"];cert_cases=pack["splits"][role]["cert"]
        heartbeat("tokenizing_training",cases=len(train_cases))
        examples=[training_example(tokenizer,cand,pack,builder,c,maxlen) for c in train_cases]
        lcfg=LoraConfig(r=int(cfg["training"]["lora_rank"]),lora_alpha=int(cfg["training"]["lora_alpha"]),lora_dropout=float(cfg["training"]["lora_dropout"]),bias="none",target_modules=cand["lora_target_regex"])
        model=get_peft_model(model,lcfg)
        for n,p in model.named_parameters():
            if "lora_" not in n: p.requires_grad_(False)
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable: raise RuntimeError("LoRA injection produced zero trainable parameters")
        if role in {"diagnosis","planner"} and any("vision_tower" in n for n,_ in trainable): raise RuntimeError("vision tower became trainable")
        opt=torch.optim.AdamW([p for _,p in trainable],lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]))
        steps=int(cfg["training"]["optimizer_steps_by_role"][role]);acc=int(cfg["training"]["gradient_accumulation_steps"])
        if steps*acc!=len(examples): raise RuntimeError(f"coverage mismatch steps*acc={steps*acc} train={len(examples)}")
        rng=random.Random(int(cfg["training"]["seed"]));order=list(range(len(examples)));rng.shuffle(order)
        checkpoints=set(int(x) for x in cfg["training"]["dev_checkpoints_by_role"][role])
        cursor=0;losses=[];supervised=nonpad=0;dev_records=[];best=None
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();train_start=time.perf_counter()
        set_training_mode(model,True)
        for step in range(1,steps+1):
            if time.monotonic()>=deadline: raise TimeoutError("role mastery hard wall during training")
            opt.zero_grad(set_to_none=True);sl=0.0
            for _ in range(acc):
                ex=examples[order[cursor]];cursor+=1
                batch={k:v.to("cuda",non_blocking=True) for k,v in ex.items() if k in {"input_ids","attention_mask","labels"}}
                out=model(**batch,use_cache=False);loss=out.loss/acc
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
                loss.backward();sl+=float(loss.detach().cpu())*acc
                supervised+=int(ex["supervised_tokens"]);nonpad+=int(ex["nonpad_tokens"])
            torch.nn.utils.clip_grad_norm_([p for _,p in trainable],float(cfg["training"]["max_grad_norm"]));opt.step()
            losses.append(sl/acc)
            if step==1 or step%25==0 or step==steps:
                heartbeat("training",optimizer_step=step,optimizer_steps=steps,loss=losses[-1])
            if step in checkpoints:
                ck=root/"checkpoints"/f"step-{step:04d}";save_adapter(model,ck)
                set_training_mode(model,False)
                dev=evaluate(model,tokenizer,cand,pack,dev_cases,int(cfg["evaluation"]["seed"]),int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,f"{prefix}/dev/step-{step:04d}.jsonl",heartbeat,f"dev_step_{step}")
                rec={"step":step,"summary":dev,"adapter_dir":str(ck)};dev_records.append(rec)
                rank=(dev["quality_100"],dev["evidence_grounding"],-dev["hallucination_rate"],dev["exact_contract_pass_rate"])
                if best is None or rank>best["rank"]: best={"rank":rank,"step":step,"adapter_dir":str(ck),"summary":dev}
                set_training_mode(model,True)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-train_start
        if cursor!=len(examples): raise RuntimeError("not all training examples consumed exactly once")
        set_training_mode(model,False)
        if best is None: raise RuntimeError("no dev checkpoint selected")
        selected_step=int(best["step"])
        if selected_step!=steps:
            base_model=model.unload();del model;gc.collect();torch.cuda.empty_cache()
            model=PeftModel.from_pretrained(base_model,best["adapter_dir"],is_trainable=False)
            model.to("cuda");model.eval()
        cert=evaluate(model,tokenizer,cand,pack,cert_cases,int(cfg["evaluation"]["seed"]),int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,f"{prefix}/certification.jsonl",heartbeat,"certification")
        cert_result=certification(cert,cfg["certification"]["thresholds"][role])
        selected_path=Path(best["adapter_dir"]);artifact=archive_adapter(selected_path,root/"selected-adapter.tar.gz",client,f"{prefix}/selected-adapter.tar.gz")
        result.update({
          "status":"completed","selected_dev_step":selected_step,"dev_checkpoints":[{"step":x["step"],"summary":x["summary"]} for x in dev_records],
          "training":{"optimizer_steps":steps,"training_cases_consumed":cursor,"supervised_token_updates":supervised,"nonpad_token_updates":nonpad,"trainable_parameter_count":sum(p.numel() for _,p in trainable),"training_seconds":train_seconds,"peak_training_vram_bytes":int(torch.cuda.max_memory_allocated()),"loss_first":losses[0],"loss_last":losses[-1],"loss_mean":sum(losses)/len(losses)},
          "certification_summary":cert,"certification":cert_result,"selected_adapter":artifact,"completed_at_unix":time.time(),
        })
        put_json(client,f"{prefix}/result.json",result);heartbeat("complete",certification=cert_result,quality_100=cert["quality_100"],selected_dev_step=selected_step)
        print("ROLE_MASTERY_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"completed_at_unix":time.time()})
        put_json(client,f"{prefix}/result.json",result);print("ROLE_MASTERY_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);raise

if __name__=="__main__":raise SystemExit(main())
