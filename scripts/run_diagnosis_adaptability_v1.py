#!/usr/bin/env python3
"""Paired micro-LoRA adaptability experiment for Hephaestus diagnosis foundations."""
from __future__ import annotations
import gc, hashlib, importlib.util, json, os, random, shutil, sys, tarfile, time, traceback
from pathlib import Path
from typing import Any

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
import run_diagnosis_foundation_bakeoff_v1 as stage1
import run_cognitive_topology_v1 as topo
import run_diagnostic_scaling_v1 as base

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/experiments/hephaestus_diagnosis_adaptability_v1.json"

def req(k:str)->str:
    v=(os.environ.get(k) or "").strip()
    if not v: raise RuntimeError("missing required environment variable: "+k)
    return v

def put(client:Any,key:str,obj:object)->None:
    raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    client.put_object(Bucket=base.bucket(),Key=key,Body=raw)
    if base.read_s3(client,key)!=raw: raise RuntimeError("S3 readback mismatch: "+key)

def pack_from_builder(cfg:dict[str,Any])->dict[str,Any]:
    p=ROOT/cfg["pack"]["builder_path"]
    spec=importlib.util.spec_from_file_location("diag_pack",p)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import diagnosis pack builder")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pack=mod.build_pack(); mod.validate(pack)
    return pack

def canonical_sha(obj:object)->str:
    raw=(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
    return hashlib.sha256(raw).hexdigest()

def expected_answer(case:dict[str,Any])->str:
    exp=case["expected"]
    conf=(float(exp["confidence_min"])+float(exp["confidence_max"]))/2.0
    decision=exp["decision"]
    if decision=="inconclusive":
        uncertainty=["The available evidence does not isolate a causal failure domain."]
        rationale="Material evidence is insufficient to justify a causal failure-domain claim; collect more evidence before changing the system."
    elif decision=="data_quality":
        uncertainty=[]
        rationale="Controlled ablation isolates the data revision as the causal variable, so repair the data path."
    elif decision=="tokenizer":
        uncertainty=[]
        rationale="The tokenizer/model incompatibility is directly demonstrated and blocks valid scientific generation."
    elif decision=="evaluation_integrity":
        uncertainty=[]
        rationale="The evaluation protocol identity differs from the frozen baseline, so repair evaluation before interpreting quality."
    else:
        raise RuntimeError("unexpected training decision "+str(decision))
    obj={
        "decision":decision,
        "action":exp["action"],
        "primary_variable":exp["primary_variable"],
        "confidence":round(conf,3),
        "evidence_refs":list(case["allowed_evidence_refs"]),
        "uncertainties":uncertainty,
        "rationale":rationale,
    }
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"))

def template_ids(tokenizer:Any,cand:dict[str,Any],messages:list[dict[str,str]],generation_prompt:bool)->list[int]:
    kwargs=dict(cand.get("chat_template_kwargs") or {})
    if cand["load_kind"]=="mistral3":
        enc=tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True,**kwargs)
    else:
        enc=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation_prompt,return_tensors="pt",return_dict=True,**kwargs)
    ids=enc["input_ids"][0].tolist()
    return [int(x) for x in ids]

def training_example(tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cfg:dict[str,Any],case:dict[str,Any])->dict[str,Any]:
    import torch
    user=stage1.prompt(pack,case,{"contract_vocabulary":cfg["contract_vocabulary"]})
    answer=expected_answer(case)
    prompt_ids=template_ids(tokenizer,cand,[{"role":"user","content":user}],True)
    full_ids=template_ids(tokenizer,cand,[{"role":"user","content":user},{"role":"assistant","content":answer}],False)
    lcp=0
    for a,b in zip(prompt_ids,full_ids):
        if a!=b: break
        lcp+=1
    if lcp<8: raise RuntimeError(f"chat-template prefix mismatch for {cand['candidate_id']} {case['case_id']}: lcp={lcp}")
    maxlen=int(cfg["training"]["max_sequence_length"])
    if len(full_ids)>maxlen:
        raise RuntimeError(f"training example exceeds frozen sequence budget: {case['case_id']} tokens={len(full_ids)} > {maxlen}")
    pad=getattr(tokenizer,"pad_token_id",None)
    if pad is None: pad=getattr(tokenizer,"eos_token_id",None)
    if isinstance(pad,(list,tuple)): pad=pad[0] if pad else None
    if pad is None: raise RuntimeError("tokenizer has no pad/eos id")
    input_ids=full_ids+[int(pad)]*(maxlen-len(full_ids))
    attn=[1]*len(full_ids)+[0]*(maxlen-len(full_ids))
    labels=[-100]*min(lcp,len(full_ids))+full_ids[min(lcp,len(full_ids)):]
    labels=labels+[-100]*(maxlen-len(labels))
    supervised=sum(x!=-100 for x in labels)
    if supervised<4: raise RuntimeError("too few supervised tokens")
    return {
        "case_id":case["case_id"],
        "input_ids":torch.tensor([input_ids],dtype=torch.long),
        "attention_mask":torch.tensor([attn],dtype=torch.long),
        "labels":torch.tensor([labels],dtype=torch.long),
        "nonpad_tokens":len(full_ids),
        "supervised_tokens":supervised,
    }

def eval_summary(rows:list[dict[str,Any]])->dict[str,Any]:
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    roots={}
    for root in ["D1","D2","D3","D4","D5","D6"]:
        xs=[r for r in rows if r["root_case_id"]==root]
        roots[root]={
            "quality_100":mean([x["score"]["quality_100"] for x in xs]),
            "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in xs]),
            "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in xs]),
            "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in xs]),
            "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in xs]),
            "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in xs]),
        }
    return {
        "sample_count":len(rows),
        "quality_100":mean([r["score"]["quality_100"] for r in rows]),
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

def evaluate(model:Any,tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cfg:dict[str,Any],cases:list[dict[str,Any]],seed:int,deadline:float,client:Any,prefix:str,phase:str,heartbeat)->tuple[list[dict[str,Any]],dict[str,Any]]:
    rows=[]
    for ix,case in enumerate(cases,1):
        if time.monotonic()>=deadline: raise TimeoutError("adaptability hard wall during evaluation")
        heartbeat(f"{phase}_probe_started",candidate_id=cand["candidate_id"],case_id=case["case_id"],case_index=ix)
        gen=stage1.generate(model,tokenizer,cand,stage1.prompt(pack,case,{"contract_vocabulary":cfg["contract_vocabulary"]}),seed,int(cfg["evaluation"]["max_new_tokens"]),deadline)
        raw=gen.pop("raw_output"); projected=stage1.project(raw,cand); normalized,extraction=stage1.extract_complete_json(projected)
        score=topo.score_response(pack,case,normalized)
        row={"sample_version":"diagnosis-adaptability-eval.v1","phase":phase,"candidate_id":cand["candidate_id"],"case_id":case["case_id"],"root_case_id":case["root_case_id"],"condition":case["condition"],"seed":seed,"raw_output":raw,"projected_output":projected,"output":normalized,"extraction":extraction,"generation":gen,"score":score}
        rows.append(row); put(client,f"{prefix}/samples/{cand['candidate_id']}/{phase}/{ix:03d}-{case['case_id']}.json",row)
        heartbeat(f"{phase}_probe_complete",candidate_id=cand["candidate_id"],case_id=case["case_id"],case_index=ix,quality_100=score["quality_100"],schema_compliant=score["schema_compliant"])
    return rows,eval_summary(rows)

def train_lora(model:Any,tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cfg:dict[str,Any],cases:list[dict[str,Any]],deadline:float,heartbeat)->tuple[Any,dict[str,Any]]:
    import torch
    from peft import LoraConfig,get_peft_model
    tr=cfg["training"]
    examples=[training_example(tokenizer,cand,pack,cfg,c) for c in cases]
    lora=LoraConfig(r=int(tr["lora_rank"]),lora_alpha=int(tr["lora_alpha"]),lora_dropout=float(tr["lora_dropout"]),bias="none",target_modules=cand["lora_target_regex"])
    model=get_peft_model(model,lora)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("LoRA injection produced zero trainable parameters")
    if cand["candidate_id"].startswith("ministral") and any("vision_tower" in n for n,_ in trainable):
        raise RuntimeError("vision tower accidentally received trainable LoRA parameters")
    for n,p in model.named_parameters():
        if "lora_" not in n: p.requires_grad_(False)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for _,p in trainable)
    matched_modules=sorted({n.rsplit(".lora_",1)[0] for n,_ in trainable if ".lora_" in n})
    if not matched_modules: raise RuntimeError("could not identify matched LoRA modules")
    if tr.get("gradient_checkpointing"):
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
        if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"): model.config.use_cache=False
    model.train()
    opt=torch.optim.AdamW([p for _,p in trainable],lr=float(tr["learning_rate"]),weight_decay=float(tr["weight_decay"]))
    rng=random.Random(int(tr["seed"]))
    order=[]
    while len(order)<int(tr["optimizer_steps"])*int(tr["gradient_accumulation_steps"]):
        epoch=list(range(len(examples))); rng.shuffle(epoch); order.extend(epoch)
    order=order[:int(tr["optimizer_steps"])*int(tr["gradient_accumulation_steps"])]
    losses=[]; supervised_updates=0; nonpad_updates=0; slots=0; micro_cursor=0
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    for step in range(1,int(tr["optimizer_steps"])+1):
        if time.monotonic()>=deadline: raise TimeoutError("adaptability hard wall during training")
        opt.zero_grad(set_to_none=True); step_loss=0.0
        for _ in range(int(tr["gradient_accumulation_steps"])):
            ex=examples[order[micro_cursor]];micro_cursor+=1
            batch={k:v.to("cuda",non_blocking=True) for k,v in ex.items() if k in {"input_ids","attention_mask","labels"}}
            out=model(**batch,use_cache=False)
            loss=out.loss/int(tr["gradient_accumulation_steps"])
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at optimizer step {step}")
            loss.backward();step_loss+=float(loss.detach().cpu())*int(tr["gradient_accumulation_steps"])
            supervised_updates+=int(ex["supervised_tokens"]);nonpad_updates+=int(ex["nonpad_tokens"]);slots+=int(tr["max_sequence_length"])
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],float(tr["max_grad_norm"]))
        opt.step();losses.append(step_loss/int(tr["gradient_accumulation_steps"]))
        heartbeat("training_step",candidate_id=cand["candidate_id"],optimizer_step=step,optimizer_steps=int(tr["optimizer_steps"]),loss=losses[-1])
    torch.cuda.synchronize();finished=time.perf_counter()
    if slots!=int(tr["padded_training_token_slots"]): raise RuntimeError(f"training slot budget mismatch: {slots}")
    if hasattr(model,"gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    model.eval()
    stats={
        "optimizer_steps":int(tr["optimizer_steps"]),
        "gradient_accumulation_steps":int(tr["gradient_accumulation_steps"]),
        "padded_training_token_slots":slots,
        "supervised_token_updates":supervised_updates,
        "nonpad_token_updates":nonpad_updates,
        "trainable_parameter_count":int(trainable_count),
        "matched_lora_module_count":len(matched_modules),
        "matched_lora_modules":matched_modules,
        "training_seconds":finished-started,
        "peak_training_vram_bytes":int(torch.cuda.max_memory_allocated()),
        "loss_first":losses[0] if losses else None,
        "loss_last":losses[-1] if losses else None,
        "loss_mean":sum(losses)/len(losses) if losses else None,
    }
    return model,stats

def save_adapter(model:Any,root:Path,client:Any,key:str)->dict[str,Any]:
    out=root/"adapter";out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out,safe_serialization=True)
    tar_path=root/"adapter.tar.gz"
    with tarfile.open(tar_path,"w:gz") as tf: tf.add(out,arcname="adapter")
    h=hashlib.sha256()
    with tar_path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""): h.update(chunk)
    client.upload_file(str(tar_path),base.bucket(),key)
    head=client.head_object(Bucket=base.bucket(),Key=key)
    return {"s3_key":key,"sha256":h.hexdigest(),"bytes":int(head["ContentLength"])}

def compare(pre:dict[str,Any],post:dict[str,Any],train:dict[str,Any])->dict[str,Any]:
    gain=post["quality_100"]-pre["quality_100"]
    root_deltas={r:post["roots"][r]["quality_100"]-pre["roots"][r]["quality_100"] for r in pre["roots"]}
    mastered=[r for r in pre["roots"] if pre["roots"][r]["quality_100"]>=90.0]
    regressions={r:root_deltas[r] for r in mastered if root_deltas[r]<0}
    sup=max(1,int(train["supervised_token_updates"]));secs=max(1e-9,float(train["training_seconds"]))
    return {
        "held_out_quality_100_pre":pre["quality_100"],
        "held_out_quality_100_post":post["quality_100"],
        "quality_gain":gain,
        "gain_per_1000_supervised_tokens":gain/(sup/1000.0),
        "gain_per_optimizer_step":gain/max(1,int(train["optimizer_steps"])),
        "gain_per_training_minute":gain/(secs/60.0),
        "causal_restraint_pre_post":{"pre":pre["roots"]["D1"]["quality_100"],"post":post["roots"]["D1"]["quality_100"]},
        "evidence_grounding_pre_post":{"pre":pre["evidence_grounding"],"post":post["evidence_grounding"]},
        "d5_admissibility_pre_post":{"pre":pre["roots"]["D5"]["evidence_grounding"],"post":post["roots"]["D5"]["evidence_grounding"]},
        "d6_evaluation_integrity_pre_post":{"pre":pre["roots"]["D6"]["quality_100"],"post":post["roots"]["D6"]["quality_100"]},
        "confidence_calibration_pre_post":{"pre":pre["confidence_calibration"],"post":post["confidence_calibration"]},
        "schema_compliance_pre_post":{"pre":pre["schema_compliance"],"post":post["schema_compliance"]},
        "exact_contract_pass_pre_post":{"pre":pre["exact_contract_pass_rate"],"post":post["exact_contract_pass_rate"]},
        "latency_pre_post":{"pre":pre["mean_latency_seconds"],"post":post["mean_latency_seconds"]},
        "root_quality_deltas":root_deltas,
        "mastered_pre_roots":mastered,
        "mastered_capability_regressions":regressions,
    }

def main()->int:
    cfg=json.loads(CFG_PATH.read_text());pack=pack_from_builder(cfg);pack_sha=canonical_sha(pack)
    if pack_sha!=cfg["parent_zero_shot"]["pack_sha256"]: raise RuntimeError("frozen diagnosis pack hash drift")
    run_id=req("HEPHAESTUS_DIAG_ADAPT_RUN_ID");repo_sha=req("HEPHAESTUS_REPO_SHA")
    client=base.s3_client();prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    parent=json.loads(base.read_s3(client,cfg["parent_zero_shot"]["result_key"]))
    if parent.get("status")!="completed" or parent.get("completed_samples")!=120 or parent.get("pack_sha256")!=pack_sha:
        raise RuntimeError("parent zero-shot evidence is incomplete or has wrong pack hash")
    deadline=time.monotonic()+int(cfg["execution"]["hard_wall_seconds"]);seed=int(cfg["training"]["seed"])
    train_cases=pack["partitions"][cfg["pack"]["train_partition"]];held=pack["partitions"][cfg["pack"]["held_out_partition"]]
    if len(train_cases)!=72 or len(held)!=48: raise RuntimeError("partition cardinality drift")
    completed_eval=0
    result={"result_version":"hephaestus-diagnosis-adaptability.v1","status":"running","run_id":run_id,"repo_sha":repo_sha,"pack_sha256":pack_sha,"parent_run_id":cfg["parent_zero_shot"]["run_id"],"candidate_results":{},"started_at_unix":time.time()}
    def heartbeat(stage:str,**extra):
        p={"run_id":run_id,"stage":stage,"completed_eval_samples":completed_eval,"expected_eval_samples":192,"timestamp_unix":time.time(),"remaining_wall_seconds":max(0,deadline-time.monotonic()),**extra}
        put(client,f"{prefix}/progress.json",p);print("DIAG_ADAPT_HEARTBEAT_JSON "+json.dumps(p,sort_keys=True),flush=True)
    put(client,f"{prefix}/run_manifest.json",{"protocol":cfg,"repo_sha":repo_sha,"pack_sha256":pack_sha,"parent_result":parent})
    try:
        for ci,cand in enumerate(cfg["candidates"],1):
            root=Path("/opt/hephaestus-diagnosis-adaptability")/run_id/cand["candidate_id"]
            heartbeat("materializing_model",candidate_id=cand["candidate_id"],candidate_index=ci)
            snap,adapter,manifest=stage1.materialize(cand,root/"model");put(client,f"{prefix}/models/{cand['candidate_id']}/model_manifest.json",manifest)
            heartbeat("loading_model",candidate_id=cand["candidate_id"],candidate_index=ci)
            model,tokenizer,runtime=stage1.load(cand,snap,adapter,{"execution":{"minimum_gpu_memory_gib":cfg["execution"]["minimum_gpu_memory_gib"]}})
            put(client,f"{prefix}/models/{cand['candidate_id']}/runtime.json",runtime)
            pre_rows,pre=evaluate(model,tokenizer,cand,pack,cfg,held,seed,deadline,client,prefix,"pre",heartbeat);completed_eval+=len(pre_rows)
            put(client,f"{prefix}/models/{cand['candidate_id']}/pre_summary.json",pre)
            heartbeat("training_started",candidate_id=cand["candidate_id"])
            model,train=train_lora(model,tokenizer,cand,pack,cfg,train_cases,deadline,heartbeat)
            put(client,f"{prefix}/models/{cand['candidate_id']}/training_summary.json",train)
            artifact=save_adapter(model,root,client,f"{prefix}/models/{cand['candidate_id']}/adapter.tar.gz")
            put(client,f"{prefix}/models/{cand['candidate_id']}/adapter_manifest.json",artifact)
            heartbeat("post_evaluation_started",candidate_id=cand["candidate_id"])
            post_rows,post=evaluate(model,tokenizer,cand,pack,cfg,held,seed,deadline,client,prefix,"post",heartbeat);completed_eval+=len(post_rows)
            put(client,f"{prefix}/models/{cand['candidate_id']}/post_summary.json",post)
            cmp=compare(pre,post,train)
            rec={"candidate_id":cand["candidate_id"],"pre":pre,"training":train,"post":post,"comparison":cmp,"adapter":artifact}
            result["candidate_results"][cand["candidate_id"]]=rec;put(client,f"{prefix}/models/{cand['candidate_id']}/result.json",rec)
            heartbeat("candidate_complete",candidate_id=cand["candidate_id"],comparison=cmp)
            del model,tokenizer;gc.collect()
            import torch
            torch.cuda.empty_cache();shutil.rmtree(root,ignore_errors=True)
        ranked=sorted(result["candidate_results"],key=lambda x:(result["candidate_results"][x]["comparison"]["gain_per_1000_supervised_tokens"],result["candidate_results"][x]["post"]["quality_100"]),reverse=True)
        result.update({"status":"completed","completed_at_unix":time.time(),"completed_eval_samples":completed_eval,"expected_eval_samples":192,"ranking_by_primary_metric":ranked,"training_performed":True,"automatic_promotion_performed":False})
        put(client,f"{prefix}/result.json",result);heartbeat("complete",ranking_by_primary_metric=ranked)
        print("DIAG_ADAPT_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update({"status":"failed","completed_at_unix":time.time(),"completed_eval_samples":completed_eval,"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        put(client,f"{prefix}/result.json",result);print("DIAG_ADAPT_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);raise

if __name__=="__main__": raise SystemExit(main())
