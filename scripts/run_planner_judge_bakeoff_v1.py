#!/usr/bin/env python3
"""Run the frozen Planner/Judge sub-10B bakeoff: zero-shot -> matched micro-LoRA -> held-out."""
from __future__ import annotations
import gc,hashlib,importlib.util,json,os,random,shutil,tarfile,time,traceback
from pathlib import Path
from typing import Any

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in os.sys.path: os.sys.path.insert(0,str(SCRIPTS))
import run_cognitive_topology_v1 as topo
import run_diagnosis_foundation_bakeoff_v1 as modelio
import run_diagnostic_scaling_v1 as base

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/experiments/hephaestus_planner_judge_bakeoff_v1.json"

ROLE_RULES={
 "planner":"Propose but never execute. Prefer exactly one primary variable, use evidence and prior-run memory, avoid known dead ends, repair invalid evaluation before training, preserve rollback and approval boundaries, and choose the highest-value controlled next intervention.",
 "judge":"Apply finite Hephaestus Judge semantics. Deterministic gates, provenance, evidence completeness, certification state, stage policy, and matching approval outrank aggregate quality or stakeholder pressure. Promotion is never inferred from vibes.",
}

def req(k:str)->str:
    v=(os.environ.get(k) or "").strip()
    if not v: raise RuntimeError("missing required environment variable: "+k)
    return v

def put(client:Any,key:str,obj:object)->None:
    raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    client.put_object(Bucket=base.bucket(),Key=key,Body=raw)
    if base.read_s3(client,key)!=raw: raise RuntimeError("S3 readback mismatch: "+key)

def pack_from_builder(cfg:dict[str,Any])->tuple[Any,dict[str,Any]]:
    p=ROOT/cfg["pack"]["builder_path"]
    s=importlib.util.spec_from_file_location("pjpack",p)
    if s is None or s.loader is None: raise RuntimeError("cannot import bakeoff pack builder")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    pack=m.build_pack();m.validate(pack)
    return m,pack

def runtime_candidate(cand:dict[str,Any],role:str)->dict[str,Any]:
    rcfg=cand["roles"][role]
    return {
      **cand,
      "license":cand["observed_hf_license"],
      "chat_template_kwargs":dict(rcfg.get("chat_template_kwargs") or {}),
      "decoding":dict(rcfg["decoding"]),
      "reasoning_projection":list(rcfg.get("reasoning_projection") or []),
    }

def prompt(pack:dict[str,Any],case:dict[str,Any])->str:
    role=case["role"];v=pack["contract_vocabulary"][role]
    evidence="\n".join(f"- {x['ref']}: {x['fact']}" for x in case["evidence"])
    allowed=", ".join(case["allowed_evidence_refs"])
    keys=", ".join(pack["response_schema"]["required_exact_keys"])
    return f"""You are acting only as the Hephaestus {role.upper()} role.

ROLE BOUNDARY:
{ROLE_RULES[role]}

SITUATION:
{case['situation']}

EVIDENCE:
{evidence}

Return exactly one JSON object and nothing else. Do not use markdown or code fences.
The object must contain exactly these keys: {keys}.
Use the global Hephaestus contract vocabulary exactly; do not paraphrase enum values:
- decision MUST be one of: {', '.join(v['decision'])}
- action MUST be one of: {', '.join(v['action'])}
- primary_variable MUST be one of: {', '.join(v['primary_variable'])}
- confidence: JSON number from 0 to 1
- evidence_refs: JSON array. Cite only material evidence; relevant refs are a subset of: {allowed}
- uncertainties: JSON array of short strings; use [] only when no material uncertainty remains
- rationale: one concise string; do not invent evidence

The vocabulary list is global and does not reveal the correct value for this case.
Do not reveal hidden reasoning. Give only the requested decision record."""

def target_answer(case:dict[str,Any])->str:
    exp=case["expected"];conf=round((float(exp["confidence_min"])+float(exp["confidence_max"]))/2,3)
    uncertainty=[]
    if case["root_case_id"] in {"P1","J2","J4","J7"}:
        uncertainty=["The blocking evidence or governance condition must be resolved before a stronger action."]
    rationale={
      "P1":"Required scientific evidence is incomplete, so planning must request a recheck before a training intervention.",
      "P2":"Controlled evidence isolates preprocessing; change only that primary variable in a reversible branch.",
      "P3":"The exact prior replacement is a known dead end; admit a different model family before branching.",
      "P4":"The comparison is invalid until the frozen evaluation protocol is repaired.",
      "P5":"Verified undertraining with improving loss supports a bounded continuation from the resumable checkpoint.",
      "P6":"The newest checkpoint is corrupt while a verified stable checkpoint exists, so rollback is the controlled action.",
      "P7":"The coverage diagnosis and admitted dataset support one controlled dataset-mixture branch.",
      "P8":"Tokenizer incompatibility is proven; propose a tokenizer branch while preserving the required approval boundary.",
      "J1":"A frozen deterministic regression blocks the candidate regardless of aggregate quality.",
      "J2":"Promotion evidence is sufficient but the required matching approval is absent.",
      "J3":"Quality, deterministic evidence, provenance, repeatability, certification, and matching approval all support promotion.",
      "J4":"Required evaluation evidence is incomplete, so retain the safer lineage.",
      "J5":"Missing immutable checkpoint provenance invalidates a promotion-capable comparison.",
      "J6":"Repeated candidate failure plus a verified stable checkpoint supports rollback.",
      "J7":"High variance and insufficient repeatability block promotion even though the mean improved.",
      "J8":"The active stage action boundary disallows promotion and requires continuation from the candidate checkpoint.",
    }[case["root_case_id"]]
    return json.dumps({
      "decision":exp["decision"],"action":exp["action"],"primary_variable":exp["primary_variable"],
      "confidence":conf,"evidence_refs":list(case["allowed_evidence_refs"]),
      "uncertainties":uncertainty,"rationale":rationale,
    },ensure_ascii=False,separators=(",",":"))

def template_ids(tokenizer:Any,cand:dict[str,Any],messages:list[dict[str,str]],generation_prompt:bool)->list[int]:
    kwargs=dict(cand.get("chat_template_kwargs") or {})
    if cand["load_kind"]=="mistral3":
        cont=bool(messages and messages[-1].get("role")=="assistant" and not generation_prompt)
        enc=tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True,continue_final_message=cont,**kwargs)
    else:
        enc=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation_prompt,return_tensors="pt",return_dict=True,**kwargs)
    return [int(x) for x in enc["input_ids"][0].tolist()]

def training_example(tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],case:dict[str,Any],maxlen:int)->dict[str,Any]:
    import torch
    user=prompt(pack,case);answer=target_answer(case)
    p=template_ids(tokenizer,cand,[{"role":"user","content":user}],True)
    f=template_ids(tokenizer,cand,[{"role":"user","content":user},{"role":"assistant","content":answer}],False)
    lcp=0
    for a,b in zip(p,f):
        if a!=b: break
        lcp+=1
    if lcp<8: raise RuntimeError(f"chat-template prefix mismatch: {cand['candidate_id']} {case['case_id']} lcp={lcp}")
    if len(f)>maxlen: raise RuntimeError(f"training example exceeds max_sequence_length: {case['case_id']} {len(f)}>{maxlen}")
    pad=getattr(tokenizer,"pad_token_id",None)
    if pad is None: pad=getattr(tokenizer,"eos_token_id",None)
    if isinstance(pad,(list,tuple)): pad=pad[0] if pad else None
    if pad is None: raise RuntimeError("tokenizer has no pad/eos id")
    ids=f+[int(pad)]*(maxlen-len(f));mask=[1]*len(f)+[0]*(maxlen-len(f))
    labels=[-100]*min(lcp,len(f))+f[min(lcp,len(f)):]
    labels+= [-100]*(maxlen-len(labels))
    return {"input_ids":torch.tensor([ids]),"attention_mask":torch.tensor([mask]),"labels":torch.tensor([labels]),"nonpad_tokens":len(f),"supervised_tokens":sum(x!=-100 for x in labels)}

def eval_summary(rows:list[dict[str,Any]])->dict[str,Any]:
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    roots={}
    for root in sorted({r["root_case_id"] for r in rows}):
        xs=[r for r in rows if r["root_case_id"]==root]
        roots[root]={
          "skill":xs[0]["skill"],"quality_100":mean([x["score"]["quality_100"] for x in xs]),
          "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in xs]),
          "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in xs]),
          "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in xs]),
          "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in xs]),
          "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in xs]),
        }
    return {
      "sample_count":len(rows),"quality_100":mean([r["score"]["quality_100"] for r in rows]),
      "schema_compliance":mean([float(r["score"]["schema_compliant"]) for r in rows]),
      "evidence_grounding":mean([r["score"]["components"]["evidence_grounding"] for r in rows]),
      "confidence_calibration":mean([r["score"]["components"]["confidence_calibration"] for r in rows]),
      "hallucination_rate":mean([r["score"]["hallucination_rate"] for r in rows]),
      "exact_contract_pass_rate":mean([float(r["score"]["quality_100"]>=99.999) for r in rows]),
      "mean_latency_seconds":mean([r["generation"]["total_latency_seconds"] for r in rows]),
      "mean_generated_tokens":mean([r["generation"]["generated_tokens"] for r in rows]),
      "peak_eval_vram_bytes":max([r["generation"]["peak_vram_bytes"] for r in rows],default=0),
      "roots":roots,
    }

def evaluate(model:Any,tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cases:list[dict[str,Any]],seed:int,max_new:int,deadline:float,client:Any,prefix:str,role:str,phase:str,heartbeat)->tuple[list[dict[str,Any]],dict[str,Any]]:
    rows=[]
    for ix,case in enumerate(cases,1):
        if time.monotonic()>=deadline: raise TimeoutError("bakeoff hard wall during evaluation")
        heartbeat(f"{phase}_probe_started",role=role,candidate_id=cand["candidate_id"],case_id=case["case_id"],case_index=ix)
        gen=modelio.generate(model,tokenizer,cand,prompt(pack,case),seed,max_new,deadline)
        raw=gen.pop("raw_output");projected=modelio.project(raw,cand);normalized,extract=modelio.extract_complete_json(projected)
        score=topo.score_response(pack,case,normalized)
        row={"sample_version":"planner-judge-bakeoff.v1","role":role,"phase":phase,"candidate_id":cand["candidate_id"],"case_id":case["case_id"],"root_case_id":case["root_case_id"],"skill":case["skill"],"condition":case["condition"],"seed":seed,"raw_output":raw,"projected_output":projected,"output":normalized,"extraction":extract,"generation":gen,"score":score}
        rows.append(row);put(client,f"{prefix}/samples/{role}/{cand['candidate_id']}/{phase}/{ix:03d}-{case['case_id']}.json",row)
        heartbeat(f"{phase}_probe_complete",role=role,candidate_id=cand["candidate_id"],case_id=case["case_id"],case_index=ix,quality_100=score["quality_100"])
    return rows,eval_summary(rows)

def train_lora(model:Any,tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cases:list[dict[str,Any]],tr:dict[str,Any],deadline:float,heartbeat,role:str)->tuple[Any,dict[str,Any]]:
    import torch
    from peft import LoraConfig,get_peft_model
    maxlen=int(tr["max_sequence_length"]);examples=[training_example(tokenizer,cand,pack,c,maxlen) for c in cases]
    lcfg=LoraConfig(r=int(tr["lora_rank"]),lora_alpha=int(tr["lora_alpha"]),lora_dropout=float(tr["lora_dropout"]),bias="none",target_modules=cand["lora_target_regex"])
    model=get_peft_model(model,lcfg)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("LoRA injection produced zero trainable parameters")
    if cand["candidate_id"].startswith("ministral") and any("vision_tower" in n for n,_ in trainable): raise RuntimeError("vision tower became trainable")
    for n,p in model.named_parameters():
        if "lora_" not in n: p.requires_grad_(False)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if tr["gradient_checkpointing"]:
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
        if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"): model.config.use_cache=False
    opt=torch.optim.AdamW([p for _,p in trainable],lr=float(tr["learning_rate"]),weight_decay=float(tr["weight_decay"]))
    rng=random.Random(int(tr["seed"]));order=[]
    needed=int(tr["optimizer_steps"])*int(tr["gradient_accumulation_steps"])
    while len(order)<needed:
        epoch=list(range(len(examples)));rng.shuffle(epoch);order.extend(epoch)
    order=order[:needed];cursor=0;losses=[];sup=nonpad=slots=0
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    model.train()
    for step in range(1,int(tr["optimizer_steps"])+1):
        if time.monotonic()>=deadline: raise TimeoutError("bakeoff hard wall during training")
        opt.zero_grad(set_to_none=True);sl=0.0
        for _ in range(int(tr["gradient_accumulation_steps"])):
            ex=examples[order[cursor]];cursor+=1
            batch={k:v.to("cuda",non_blocking=True) for k,v in ex.items() if k in {"input_ids","attention_mask","labels"}}
            out=model(**batch,use_cache=False);loss=out.loss/int(tr["gradient_accumulation_steps"])
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward();sl+=float(loss.detach().cpu())*int(tr["gradient_accumulation_steps"])
            sup+=int(ex["supervised_tokens"]);nonpad+=int(ex["nonpad_tokens"]);slots+=maxlen
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],float(tr["max_grad_norm"]));opt.step()
        losses.append(sl/int(tr["gradient_accumulation_steps"]))
        heartbeat("training_step",role=role,candidate_id=cand["candidate_id"],optimizer_step=step,optimizer_steps=int(tr["optimizer_steps"]),loss=losses[-1])
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    if slots!=int(tr["padded_training_token_slots_per_pair"]): raise RuntimeError(f"slot budget mismatch {slots}")
    if hasattr(model,"gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    model.eval()
    mods=sorted({n.rsplit(".lora_",1)[0] for n,_ in trainable if ".lora_" in n})
    return model,{"optimizer_steps":int(tr["optimizer_steps"]),"padded_training_token_slots":slots,"supervised_token_updates":sup,"nonpad_token_updates":nonpad,"trainable_parameter_count":sum(p.numel() for _,p in trainable),"matched_lora_module_count":len(mods),"matched_lora_modules":mods,"training_seconds":seconds,"peak_training_vram_bytes":int(torch.cuda.max_memory_allocated()),"loss_first":losses[0],"loss_last":losses[-1],"loss_mean":sum(losses)/len(losses)}

def save_adapter(model:Any,root:Path,client:Any,key:str)->dict[str,Any]:
    out=root/"adapter";out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out,safe_serialization=True)
    tar=root/"adapter.tar.gz"
    with tarfile.open(tar,"w:gz") as tf: tf.add(out,arcname="adapter")
    h=hashlib.sha256()
    with tar.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):h.update(chunk)
    client.upload_file(str(tar),base.bucket(),key);head=client.head_object(Bucket=base.bucket(),Key=key)
    return {"s3_key":key,"sha256":h.hexdigest(),"bytes":int(head["ContentLength"])}

def comparison(zero:dict[str,Any],pre:dict[str,Any],post:dict[str,Any],train:dict[str,Any])->dict[str,Any]:
    gain=post["quality_100"]-pre["quality_100"];sup=max(1,train["supervised_token_updates"]);sec=max(1e-9,train["training_seconds"])
    mastered=[r for r,v in pre["roots"].items() if v["quality_100"]>=90]
    deltas={r:post["roots"][r]["quality_100"]-pre["roots"][r]["quality_100"] for r in pre["roots"]}
    return {
      "zero_shot_quality_100":zero["quality_100"],"held_out_quality_100_pre":pre["quality_100"],"held_out_quality_100_post":post["quality_100"],
      "quality_gain":gain,"gain_per_1000_supervised_tokens":gain/(sup/1000.0),"gain_per_optimizer_step":gain/max(1,train["optimizer_steps"]),"gain_per_training_minute":gain/(sec/60),
      "schema_compliance_pre_post":{"pre":pre["schema_compliance"],"post":post["schema_compliance"]},
      "evidence_grounding_pre_post":{"pre":pre["evidence_grounding"],"post":post["evidence_grounding"]},
      "hallucination_pre_post":{"pre":pre["hallucination_rate"],"post":post["hallucination_rate"]},
      "confidence_calibration_pre_post":{"pre":pre["confidence_calibration"],"post":post["confidence_calibration"]},
      "exact_contract_pass_pre_post":{"pre":pre["exact_contract_pass_rate"],"post":post["exact_contract_pass_rate"]},
      "latency_pre_post":{"pre":pre["mean_latency_seconds"],"post":post["mean_latency_seconds"]},
      "root_skill_pre_post":{r:{"skill":pre["roots"][r]["skill"],"pre":pre["roots"][r]["quality_100"],"post":post["roots"][r]["quality_100"],"delta":deltas[r]} for r in pre["roots"]},
      "mastered_pre_roots":mastered,"mastered_capability_regressions":{r:deltas[r] for r in mastered if deltas[r]<0},
    }

def main()->int:
    cfg=json.loads(CFG_PATH.read_text());builder,pack=pack_from_builder(cfg);pack_sha=builder.canonical_sha256(pack)
    if pack_sha!=cfg["pack"]["canonical_sha256"]: raise RuntimeError("frozen pack hash drift")
    run_id=req("HEPHAESTUS_PJ_BAKEOFF_RUN_ID");repo_sha=req("HEPHAESTUS_REPO_SHA");client=base.s3_client()
    prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}";deadline=time.monotonic()+int(cfg["execution"]["hard_wall_seconds"])
    registry={x["candidate_id"]:x for x in cfg["candidate_registry"]};seed=int(cfg["training"]["seed"]);completed=0;expected_eval=8*(32+48+48)
    result={"result_version":"hephaestus-planner-judge-bakeoff.v1","status":"running","run_id":run_id,"repo_sha":repo_sha,"pack_sha256":pack_sha,"pair_results":{},"started_at_unix":time.time(),"automatic_selection_performed":False,"promotion_performed":False}
    def heartbeat(stage:str,**extra):
        p={"run_id":run_id,"stage":stage,"completed_eval_samples":completed,"expected_eval_samples":expected_eval,"timestamp_unix":time.time(),"remaining_wall_seconds":max(0,deadline-time.monotonic()),**extra}
        put(client,f"{prefix}/progress.json",p);print("PJ_BAKEOFF_HEARTBEAT_JSON "+json.dumps(p,sort_keys=True),flush=True)
    put(client,f"{prefix}/run_manifest.json",{"protocol":cfg,"pack_sha256":pack_sha,"repo_sha":repo_sha})
    try:
        for role in ("planner","judge"):
            for ci,cid in enumerate(cfg["role_rosters"][role],1):
                pair=f"{role}--{cid}";pair_key=f"{prefix}/pairs/{pair}/result.json"
                try: existing=json.loads(base.read_s3(client,pair_key))
                except Exception: existing=None
                if isinstance(existing,dict) and existing.get("status")=="complete":
                    result["pair_results"][pair]=existing;completed+=128;heartbeat("pair_resume_skip",role=role,candidate_id=cid);continue
                cand=runtime_candidate(registry[cid],role);root=Path("/opt/hephaestus-pj-bakeoff")/run_id/pair
                heartbeat("materializing_model",role=role,candidate_id=cid,pair_index=ci)
                snap,adapter,manifest=modelio.materialize(cand,Path("/opt/hephaestus-pj-bakeoff")/run_id/"shared"/cid/"model")
                put(client,f"{prefix}/models/{cid}/manifest.json",manifest)
                model,tokenizer,runtime=modelio.load(cand,snap,adapter,cfg);put(client,f"{prefix}/pairs/{pair}/runtime.json",runtime)
                zrows,zero=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["zero_shot"],seed,int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,prefix,role,"zero_shot",heartbeat);completed+=len(zrows)
                prows,pre=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["held_out"],seed,int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,prefix,role,"pre",heartbeat);completed+=len(prows)
                heartbeat("training_started",role=role,candidate_id=cid)
                model,train=train_lora(model,tokenizer,cand,pack,pack["partitions"][role]["micro_lora_train"],cfg["training"],deadline,heartbeat,role)
                artifact=save_adapter(model,root,client,f"{prefix}/pairs/{pair}/adapter.tar.gz")
                orows,post=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["held_out"],seed,int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,prefix,role,"post",heartbeat);completed+=len(orows)
                comp=comparison(zero,pre,post,train)
                rec={"status":"complete","role":role,"candidate_id":cid,"model_id":cand["model_id"],"revision":cand["revision"],"zero_shot":zero,"pre":pre,"training":train,"post":post,"comparison":comp,"adapter":artifact}
                result["pair_results"][pair]=rec;put(client,pair_key,rec);heartbeat("pair_complete",role=role,candidate_id=cid,comparison=comp)
                del model,tokenizer;gc.collect()
                import torch
                torch.cuda.empty_cache();shutil.rmtree(root,ignore_errors=True)
        rankings={}
        for role in ("planner","judge"):
            rows=[v for v in result["pair_results"].values() if v["role"]==role]
            rows=sorted(rows,key=lambda x:(x["post"]["quality_100"],x["comparison"]["quality_gain"],x["post"]["evidence_grounding"],-x["post"]["hallucination_rate"]),reverse=True)
            rankings[role]=[{"candidate_id":x["candidate_id"],"post_quality_100":x["post"]["quality_100"],"quality_gain":x["comparison"]["quality_gain"],"gain_per_1000_supervised_tokens":x["comparison"]["gain_per_1000_supervised_tokens"],"mastered_capability_regressions":x["comparison"]["mastered_capability_regressions"]} for x in rows]
        result.update({"status":"completed","completed_at_unix":time.time(),"completed_eval_samples":completed,"expected_eval_samples":expected_eval,"rankings_by_role":rankings,"training_performed":True})
        put(client,f"{prefix}/result.json",result);heartbeat("complete",rankings_by_role=rankings);print("PJ_BAKEOFF_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update({"status":"failed","completed_at_unix":time.time(),"completed_eval_samples":completed,"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        put(client,f"{prefix}/result.json",result);print("PJ_BAKEOFF_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);raise

if __name__=="__main__": raise SystemExit(main())
