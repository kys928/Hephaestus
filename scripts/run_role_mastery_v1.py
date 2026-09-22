#!/usr/bin/env python3
"""Phase I independent role mastery for one frozen Hephaestus role."""
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
CFG_PATH=ROOT/"configs/experiments/hephaestus_role_mastery_v1.json"

ROLE_RULES={
"controller":"Execute only the already-authorized state transition. Do not re-plan, re-diagnose, override policy, substitute a preferred action, bypass approvals, exceed budget, cross lineage boundaries, or guess around stale/corrupt state.",
"diagnosis":"Infer only what the evidence justifies. Separate observation from causality, prefer inconclusive when controls are missing, identify the most justified failure domain, calibrate confidence, and never prescribe a broader experiment than the evidence supports.",
"planner":"Propose but never execute. Choose one primary intervention at a time, use diagnosis and historical dead ends, repair invalid evaluation before training, preserve approvals and rollback, and prefer the highest-value controlled next experiment.",
"evaluator":"Interpret completed experimental evidence only. Deterministic gates, completeness, repeatability, provenance and effect size outrank narrative impressions or public benchmarks. Distinguish improvement, regression, equivalence, incompleteness and certification readiness.",
"judge":"Apply finite Hephaestus governance semantics. Hard gates, provenance, evidence completeness, certification, stage policy and matching approvals outrank aggregate quality or stakeholder pressure. Choose the permitted next transition, not a new experiment.",
}

def req(k:str)->str:
    v=(os.environ.get(k) or "").strip()
    if not v: raise RuntimeError("missing required environment variable: "+k)
    return v

def put(client:Any,key:str,obj:object)->None:
    raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    client.put_object(Bucket=base.bucket(),Key=key,Body=raw)
    if base.read_s3(client,key)!=raw: raise RuntimeError("S3 readback mismatch: "+key)

def load_pack(cfg:dict[str,Any])->tuple[Any,dict[str,Any]]:
    p=ROOT/cfg["corpus"]["builder_path"]
    s=importlib.util.spec_from_file_location("role_mastery_pack",p)
    if s is None or s.loader is None: raise RuntimeError("cannot import role mastery corpus builder")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    pack=m.build_pack();m.validate(pack)
    if m.canonical_sha256(pack)!=cfg["corpus"]["canonical_sha256"]: raise RuntimeError("frozen role mastery corpus hash drift")
    return m,pack

def candidate_for(cfg:dict[str,Any],role:str)->dict[str,Any]:
    c=dict(cfg["candidates"][role]);c["candidate_id"]=role
    return c

def prompt(pack:dict[str,Any],case:dict[str,Any],role:str)->str:
    v=pack["contract_vocabulary"][role]
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
Use the Hephaestus contract vocabulary exactly:
- decision MUST be one of: {', '.join(v['decision'])}
- action MUST be one of: {', '.join(v['action'])}
- primary_variable MUST be one of: {', '.join(v['primary_variable'])}
- confidence: JSON number from 0 to 1
- evidence_refs: JSON array containing only material evidence references, selected from: {allowed}
- uncertainties: JSON array of short strings
- rationale: one concise evidence-grounded string

Do not perform another Hephaestus role's job. Do not reveal hidden reasoning. Give only the requested decision record."""

RATIONALE={
"C1":"The requested transition is authorized against the current state and should be executed exactly.","C2":"Required promotion approval is absent, so execution must stop at the approval boundary.","C3":"Authorization is stale because the live state version changed.","C4":"The checkpoint belongs to a different lineage and cannot be used by this authorization.","C5":"The identical idempotency key can be retried because no committed transition exists.","C6":"The authorized request cannot execute because it exceeds the immutable compute budget.","C7":"Checkpoint integrity does not match the immutable manifest.","C8":"The transition already committed under this idempotency key and must not execute twice.",
"D1":"Multiple changes correlate with the regression but no controlled evidence isolates a cause.","D2":"Tokenizer/model incompatibility directly prevents generation.","D3":"The comparison is scientifically invalid until the frozen evaluation protocol is restored.","D4":"The failing capability is absent from the training mixture while other controls pass.","D5":"A controlled replay isolates wrapper serialization as the changing factor.","D6":"Improving train and validation loss without instability supports undertraining.","D7":"The newest checkpoint fails integrity while the prior checkpoint is verified.","D8":"Controlled evidence localizes the persistent capability boundary to this model family.",
"P1":"Required evidence is incomplete, so the next intervention is to collect the missing diagnostic evidence.","P2":"Controlled evidence isolates preprocessing, so only that primary variable should change.","P3":"The exact replacement is a known dead end; admit and branch to the independent family.","P4":"The evaluation comparison is invalid and must be repaired before more training.","P5":"Verified undertraining supports a bounded continuation from the resumable checkpoint.","P6":"A corrupt newest checkpoint and verified stable checkpoint justify rollback.","P7":"The admitted dataset directly addresses the diagnosed coverage gap in a reversible branch.","P8":"Tokenizer incompatibility is proven; propose the tokenizer branch while preserving approval.",
"E1":"A repeated frozen hard-gate failure scientifically rejects the candidate despite aggregate quality.","E2":"Missing required samples make the comparison incomplete regardless of partial quality.","E3":"Complete low-variance evidence with passing hard gates demonstrates genuine improvement.","E4":"Complete valid evidence shows a material consistent candidate regression.","E5":"High repeat variance prevents a stable conclusion and requires recheck.","E6":"Missing immutable provenance prevents a valid candidate conclusion.","E7":"The observed effect lies inside the declared equivalence margin.","E8":"Complete, repeatable, provenance-valid evidence satisfies the certification bundle.",
"J1":"A frozen deterministic failure blocks promotion regardless of aggregate score.","J2":"Scientific promotion criteria pass but the required matching approval is missing.","J3":"Scientific, provenance, certification and approval requirements all permit promotion.","J4":"Incomplete required evidence blocks promotion and preserves the verified lineage.","J5":"Missing immutable candidate provenance blocks promotion.","J6":"Repeated candidate failure with a verified stable checkpoint permits rollback.","J7":"High variance and insufficient repeatability block promotion.","J8":"Stage policy disallows promotion but explicitly permits continuation from the checkpoint.",
}

def target_answer(case:dict[str,Any])->str:
    e=case["expected"];root=case["root_case_id"]
    conf=round((float(e["confidence_min"])+float(e["confidence_max"]))/2,3)
    unc=[]
    if root in {"D1","P1","E2","E5","E6","J2","J4","J7","C2","C3","C4","C6","C7"}:
        unc=["The blocking uncertainty or governance condition must be resolved before a stronger action."]
    return json.dumps({"decision":e["decision"],"action":e["action"],"primary_variable":e["primary_variable"],"confidence":conf,
        "evidence_refs":list(case["allowed_evidence_refs"]),"uncertainties":unc,"rationale":RATIONALE[root]},
        ensure_ascii=False,separators=(",",":"))

def template_ids(tokenizer:Any,cand:dict[str,Any],messages:list[dict[str,str]],generation_prompt:bool)->list[int]:
    kwargs=dict(cand.get("chat_template_kwargs") or {})
    if cand["load_kind"]=="mistral3":
        cont=bool(messages and messages[-1].get("role")=="assistant" and not generation_prompt)
        enc=tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True,continue_final_message=cont,**kwargs)
    else:
        enc=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation_prompt,return_tensors="pt",return_dict=True,**kwargs)
    return [int(x) for x in enc["input_ids"][0].tolist()]

def training_example(tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],case:dict[str,Any],role:str,maxlen:int)->dict[str,Any]:
    import torch
    q=prompt(pack,case,role);a=target_answer(case)
    p=template_ids(tokenizer,cand,[{"role":"user","content":q}],True)
    f=template_ids(tokenizer,cand,[{"role":"user","content":q},{"role":"assistant","content":a}],False)
    lcp=0
    for x,y in zip(p,f):
        if x!=y: break
        lcp+=1
    if lcp<8: raise RuntimeError(f"chat-template prefix mismatch {role} {case['case_id']} lcp={lcp}")
    if len(f)>maxlen: raise RuntimeError(f"training example exceeds max length {role} {case['case_id']} {len(f)}>{maxlen}")
    pad=getattr(tokenizer,"pad_token_id",None)
    if pad is None: pad=getattr(tokenizer,"eos_token_id",None)
    if isinstance(pad,(list,tuple)): pad=pad[0] if pad else None
    if pad is None: raise RuntimeError("tokenizer has no pad/eos id")
    ids=f+[int(pad)]*(maxlen-len(f));mask=[1]*len(f)+[0]*(maxlen-len(f))
    labels=[-100]*min(lcp,len(f))+f[min(lcp,len(f)):]
    labels += [-100]*(maxlen-len(labels))
    return {"input_ids":torch.tensor([ids]),"attention_mask":torch.tensor([mask]),"labels":torch.tensor([labels]),
            "nonpad_tokens":len(f),"supervised_tokens":sum(x!=-100 for x in labels)}

def extract_json(raw:str)->tuple[str,dict[str,Any]]:
    return modelio.extract_complete_json(raw)

def summary(rows:list[dict[str,Any]])->dict[str,Any]:
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    roots={}
    for root in sorted({r["root_case_id"] for r in rows}):
        xs=[r for r in rows if r["root_case_id"]==root]
        roots[root]={"skill":xs[0]["skill"],"quality_100":mean([x["score"]["quality_100"] for x in xs]),
            "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in xs]),
            "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in xs]),
            "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in xs]),
            "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in xs]),
            "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in xs])}
    return {"sample_count":len(rows),"quality_100":mean([r["score"]["quality_100"] for r in rows]),
        "schema_compliance":mean([float(r["score"]["schema_compliant"]) for r in rows]),
        "evidence_grounding":mean([r["score"]["components"]["evidence_grounding"] for r in rows]),
        "confidence_calibration":mean([r["score"]["components"]["confidence_calibration"] for r in rows]),
        "hallucination_rate":mean([r["score"]["hallucination_rate"] for r in rows]),
        "exact_contract_pass_rate":mean([float(r["score"]["quality_100"]>=99.999) for r in rows]),
        "mean_latency_seconds":mean([r["generation"]["total_latency_seconds"] for r in rows]),
        "mean_generated_tokens":mean([r["generation"]["generated_tokens"] for r in rows]),
        "peak_eval_vram_bytes":max([r["generation"]["peak_vram_bytes"] for r in rows],default=0),"roots":roots}

def evaluate(model,tokenizer,cand,pack,cases,role,seed,max_new,deadline,client,prefix,phase,heartbeat):
    rows=[]
    for i,case in enumerate(cases,1):
        if time.monotonic()>=deadline: raise TimeoutError("role mastery hard wall during evaluation")
        gen=modelio.generate(model,tokenizer,cand,prompt(pack,case,role),seed,max_new,deadline)
        raw=gen.pop("raw_output");normalized,extraction=extract_json(raw);score=topo.score_response(pack,case,normalized)
        rec={"sample_version":"role-mastery.v1","role":role,"phase":phase,"case_id":case["case_id"],"root_case_id":case["root_case_id"],
             "skill":case["skill"],"raw_output":raw,"output":normalized,"extraction":extraction,"generation":gen,"score":score}
        rows.append(rec)
        put(client,f"{prefix}/samples/{phase}/{i:03d}-{case['case_id']}.json",rec)
        if i==1 or i%16==0 or i==len(cases): heartbeat(f"{phase}_progress",completed=i,total=len(cases),quality_100=summary(rows)["quality_100"])
    return rows,summary(rows)

def train(model,tokenizer,cand,pack,cases,role,cfg,deadline,heartbeat):
    import torch
    from peft import LoraConfig,get_peft_model
    tr=cfg["training"];maxlen=int(tr["max_sequence_length"])
    examples=[training_example(tokenizer,cand,pack,c,role,maxlen) for c in cases]
    lc=LoraConfig(r=int(tr["lora_rank"]),lora_alpha=int(tr["lora_alpha"]),lora_dropout=float(tr["lora_dropout"]),bias="none",target_modules=cand["lora_target_regex"])
    model=get_peft_model(model,lc)
    for n,p in model.named_parameters():
        if "lora_" not in n: p.requires_grad_(False)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("LoRA injection produced zero trainable parameters")
    if cand["load_kind"]=="mistral3" and any("vision_tower" in n for n,_ in trainable): raise RuntimeError("vision tower became trainable")
    if tr["gradient_checkpointing"]:
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
        if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"): model.config.use_cache=False
    opt=torch.optim.AdamW([p for _,p in trainable],lr=float(tr["learning_rate"]),weight_decay=float(tr["weight_decay"]))
    rng=random.Random(int(tr["seed"]));order=list(range(len(examples)));rng.shuffle(order)
    needed=int(tr["optimizer_steps"])*int(tr["gradient_accumulation_steps"])
    if needed!=len(examples): raise RuntimeError(f"training geometry expects exactly one shuffled epoch: needed={needed} cases={len(examples)}")
    cursor=0;losses=[];sup=nonpad=slots=0
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();model.train()
    for step in range(1,int(tr["optimizer_steps"])+1):
        if time.monotonic()>=deadline: raise TimeoutError("role mastery hard wall during training")
        opt.zero_grad(set_to_none=True);total=0.0
        for _ in range(int(tr["gradient_accumulation_steps"])):
            ex=examples[order[cursor]];cursor+=1
            batch={k:v.to("cuda",non_blocking=True) for k,v in ex.items() if k in {"input_ids","attention_mask","labels"}}
            out=model(**batch,use_cache=False);loss=out.loss/int(tr["gradient_accumulation_steps"])
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward();total+=float(loss.detach().cpu())*int(tr["gradient_accumulation_steps"])
            sup+=int(ex["supervised_tokens"]);nonpad+=int(ex["nonpad_tokens"]);slots+=maxlen
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],float(tr["max_grad_norm"]));opt.step()
        losses.append(total/int(tr["gradient_accumulation_steps"]))
        if step==1 or step%16==0 or step==int(tr["optimizer_steps"]): heartbeat("training_progress",optimizer_step=step,optimizer_steps=int(tr["optimizer_steps"]),loss=losses[-1])
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    if slots!=int(tr["padded_training_token_slots_per_role"]): raise RuntimeError(f"padded token slot mismatch {slots}")
    if hasattr(model,"gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    model.eval()
    modules=sorted({n.rsplit(".lora_",1)[0] for n,_ in trainable if ".lora_" in n})
    return model,{"optimizer_steps":int(tr["optimizer_steps"]),"padded_training_token_slots":slots,"supervised_token_updates":sup,"nonpad_token_updates":nonpad,
        "trainable_parameter_count":sum(p.numel() for _,p in trainable),"matched_lora_module_count":len(modules),"matched_lora_modules":modules,
        "training_seconds":seconds,"peak_training_vram_bytes":int(torch.cuda.max_memory_allocated()),"loss_first":losses[0],"loss_last":losses[-1],"loss_mean":sum(losses)/len(losses)}

def save_adapter(model,root:Path,client,key:str):
    out=root/"adapter";out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out,safe_serialization=True)
    tar=root/"adapter.tar.gz"
    with tarfile.open(tar,"w:gz") as tf: tf.add(out,arcname="adapter")
    h=hashlib.sha256()
    with tar.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    client.upload_file(str(tar),base.bucket(),key);head=client.head_object(Bucket=base.bucket(),Key=key)
    return {"s3_key":key,"sha256":h.hexdigest(),"bytes":int(head["ContentLength"])}

def certify(pre,post,cfg):
    g=cfg["certification"];mastered=[r for r,v in pre["roots"].items() if v["quality_100"]>=90]
    regress={r:post["roots"][r]["quality_100"]-pre["roots"][r]["quality_100"] for r in mastered if post["roots"][r]["quality_100"]<pre["roots"][r]["quality_100"]}
    checks={
      "overall_quality":post["quality_100"]>=float(g["minimum_overall_quality_100"]),
      "minimum_root_quality":min(v["quality_100"] for v in post["roots"].values())>=float(g["minimum_root_quality_100"]),
      "schema":post["schema_compliance"]>=float(g["schema_compliance_required"]),
      "grounding":post["evidence_grounding"]>=float(g["minimum_evidence_grounding"]),
      "hallucination":post["hallucination_rate"]<=float(g["maximum_hallucination_rate"]),
      "calibration":post["confidence_calibration"]>=float(g["minimum_confidence_calibration"]),
      "exact_contract":post["exact_contract_pass_rate"]>=float(g["minimum_exact_contract_pass_rate"]),
      "no_mastered_regressions":not regress,
    }
    return {"certified":all(checks.values()),"checks":checks,"mastered_pre_roots":mastered,"mastered_capability_regressions":regress}

def main()->int:
    cfg=json.loads(CFG_PATH.read_text());_,pack=load_pack(cfg);role=req("HEPHAESTUS_ROLE")
    if role not in cfg["candidates"]: raise RuntimeError("unknown frozen role: "+role)
    run_id=req("HEPHAESTUS_ROLE_MASTERY_RUN_ID");repo_sha=req("HEPHAESTUS_REPO_SHA");cand=candidate_for(cfg,role)
    client=base.s3_client();prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}/{role}"
    deadline=time.monotonic()+int(cfg["execution"]["hard_wall_seconds_per_role"])
    result={"result_version":"hephaestus-role-mastery.v1","status":"running","role":role,"run_id":run_id,"repo_sha":repo_sha,
            "corpus_sha256":cfg["corpus"]["canonical_sha256"],"model_id":cand["model_id"],"revision":cand["revision"],"promotion_performed":False}
    def heartbeat(stage,**extra):
        p={"stage":stage,"role":role,"run_id":run_id,"timestamp_unix":time.time(),"remaining_wall_seconds":max(0,deadline-time.monotonic()),**extra}
        put(client,f"{prefix}/progress.json",p);print("ROLE_MASTERY_HEARTBEAT_JSON "+json.dumps(p,sort_keys=True),flush=True)
    put(client,f"{prefix}/run_manifest.json",{"protocol":cfg,"role":role,"model":cand,"repo_sha":repo_sha})
    root=Path("/opt/hephaestus-role-mastery")/run_id/role
    try:
        heartbeat("materializing_model")
        snap,adapter,manifest=modelio.materialize(cand,root/"model");put(client,f"{prefix}/model_manifest.json",manifest)
        model,tokenizer,runtime=modelio.load(cand,snap,adapter,{"execution":{"minimum_gpu_memory_gib":cfg["execution"]["minimum_gpu_memory_gib"]}})
        put(client,f"{prefix}/runtime.json",runtime);heartbeat("pre_certification_started")
        pre_rows,pre=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["certification"],role,int(cfg["evaluation"]["seed"]),int(cand["max_new_tokens"]),deadline,client,prefix,"pre",heartbeat)
        heartbeat("training_started",pre_quality_100=pre["quality_100"])
        model,tr=train(model,tokenizer,cand,pack,pack["partitions"][role]["train"],role,cfg,deadline,heartbeat)
        artifact=save_adapter(model,root,client,f"{prefix}/adapter.tar.gz");heartbeat("post_certification_started")
        post_rows,post=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["certification"],role,int(cfg["evaluation"]["seed"]),int(cand["max_new_tokens"]),deadline,client,prefix,"post",heartbeat)
        cert=certify(pre,post,cfg)
        result.update({"status":"completed","pre":pre,"training":tr,"post":post,"quality_gain":post["quality_100"]-pre["quality_100"],"certification":cert,"adapter":artifact,"completed_at_unix":time.time()})
        put(client,f"{prefix}/result.json",result);heartbeat("complete",certified=cert["certified"],post_quality_100=post["quality_100"],quality_gain=result["quality_gain"],adapter_sha256=artifact["sha256"])
        print("ROLE_MASTERY_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"completed_at_unix":time.time()})
        put(client,f"{prefix}/result.json",result);print("ROLE_MASTERY_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);raise
    finally:
        shutil.rmtree(root,ignore_errors=True)

if __name__=="__main__": raise SystemExit(main())
