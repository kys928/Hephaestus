#!/usr/bin/env python3
"""Run the bounded zero-shot Diagnosis Foundation Bakeoff V1."""
from __future__ import annotations
import gc, hashlib, importlib.util, json, os, shutil, sys, time, traceback
from pathlib import Path
from typing import Any

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
import run_cognitive_topology_v1 as topo
import run_diagnostic_scaling_v1 as base

ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/"configs/experiments/hephaestus_diagnosis_foundation_bakeoff_v1.json"

def req(k:str)->str:
    v=(os.environ.get(k) or "").strip()
    if not v: raise RuntimeError("missing required environment variable: "+k)
    return v

def put(client:Any,key:str,obj:object)->None:
    raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    client.put_object(Bucket=base.bucket(),Key=key,Body=raw)
    if base.read_s3(client,key)!=raw: raise RuntimeError("S3 readback mismatch: "+key)

def pack_from_builder(cfg:dict[str,Any])->dict[str,Any]:
    path=ROOT/cfg["pack"]["builder_path"]
    spec=importlib.util.spec_from_file_location("diagnosis_foundation_pack",path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import pack builder")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pack=mod.build_pack(); mod.validate(pack)
    return pack

def canonical_sha(obj:object)->str:
    raw=(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
    return hashlib.sha256(raw).hexdigest()

def token_ids(value:Any)->list[int]:
    if value is None:return []
    if isinstance(value,int):return [int(value)]
    if isinstance(value,(list,tuple,set)):return sorted({int(x) for x in value})
    try:return sorted({int(x) for x in value.tolist()})
    except Exception:return []

def stop_ids(model:Any,tokenizer:Any)->list[int]:
    ids=set(token_ids(getattr(getattr(model,"generation_config",None),"eos_token_id",None)))
    ids.update(token_ids(getattr(getattr(model,"config",None),"eos_token_id",None)))
    ids.update(token_ids(getattr(tokenizer,"eos_token_id",None)))
    if not ids: raise RuntimeError("candidate has no observable EOS/turn termination IDs")
    return sorted(ids)

def project(raw:str,candidate:dict[str,Any])->str:
    best=-1
    for delim in candidate.get("reasoning_projection",[]):
        i=raw.rfind(str(delim))
        if i>=0: best=max(best,i+len(str(delim)))
    return (raw[best:] if best>=0 else raw).strip()

def materialize(candidate:dict[str,Any],root:Path)->tuple[Path,Path|None,dict[str,Any]]:
    if candidate["load_kind"]!="peft_adapter":
        snap,manifest=base.materialize_model(candidate,root)
        return snap,None,manifest
    base_candidate={"model_id":candidate["base_model_id"],"revision":candidate["base_revision"],"license":candidate["base_license"]}
    base_snap,base_manifest=base.materialize_model(base_candidate,root/"base")
    adapter_snap,adapter_manifest=base.materialize_model(candidate,root/"adapter")
    return base_snap,adapter_snap,{"base":base_manifest,"adapter":adapter_manifest}

def load(candidate:dict[str,Any],snapshot:Path,adapter:Path|None,cfg:dict[str,Any]):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("exactly one CUDA GPU required")
    props=torch.cuda.get_device_properties(0)
    if props.total_memory/(1024**3)<float(cfg["execution"]["minimum_gpu_memory_gib"]): raise RuntimeError("GPU below memory floor")
    kind=candidate["load_kind"]
    if kind=="mistral3":
        from transformers import Mistral3ForConditionalGeneration,MistralCommonBackend
        tokenizer=MistralCommonBackend.from_pretrained(str(snapshot))
        model=Mistral3ForConditionalGeneration.from_pretrained(str(snapshot),torch_dtype=torch.bfloat16,device_map={"":0},local_files_only=True)
    else:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True)
        if kind=="peft_adapter":
            if adapter is None: raise RuntimeError("PEFT adapter snapshot missing")
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    if getattr(tokenizer,"pad_token_id",None) is None and getattr(tokenizer,"eos_token_id",None) is not None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    dtypes={str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    devices={str(p.device) for p in model.parameters()}
    if not devices or not all(d.startswith("cuda") for d in devices): raise RuntimeError("candidate has non-CUDA parameter residency")
    return model,tokenizer,{"gpu":props.name,"gpu_memory_bytes":int(props.total_memory),"weight_dtypes":sorted(dtypes),"tensor_devices":sorted(devices),"allocated_memory_bytes_after_load":int(torch.cuda.memory_allocated())}

def encode(tokenizer:Any,candidate:dict[str,Any],prompt:str)->dict[str,Any]:
    messages=[{"role":"user","content":prompt}]
    kwargs=dict(candidate.get("chat_template_kwargs") or {})
    if candidate["load_kind"]=="mistral3":
        encoded=tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True,**kwargs)
    else:
        encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True,**kwargs)
    return {k:v.to("cuda") if hasattr(v,"to") else v for k,v in encoded.items()}

def generate(model:Any,tokenizer:Any,candidate:dict[str,Any],prompt:str,seed:int,max_new:int,deadline:float)->dict[str,Any]:
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    class Deadline(StoppingCriteria):
        def __init__(self):self.hit=False;self.first=None
        def __call__(self,input_ids,scores,**kw):
            now=time.monotonic()
            if self.first is None:
                torch.cuda.synchronize();self.first=time.perf_counter()
            if now>=deadline:self.hit=True;return True
            return False
    enc=encode(tokenizer,candidate,prompt); width=int(enc["input_ids"].shape[-1])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    dec=dict(candidate["decoding"]); clock=Deadline(); started=time.perf_counter()
    with torch.inference_mode():
        out=model.generate(**enc,**dec,max_new_tokens=max_new,pad_token_id=getattr(tokenizer,"pad_token_id",None),eos_token_id=stop_ids(model,tokenizer),stopping_criteria=StoppingCriteriaList([clock]),return_dict_in_generate=True,use_cache=True)
    torch.cuda.synchronize();finished=time.perf_counter()
    cont=out.sequences[0][width:]; ids=set(stop_ids(model,tokenizer)); n=int(cont.shape[0]); final=int(cont[-1]) if n else None
    raw=tokenizer.decode(cont,skip_special_tokens=True).strip()
    reason="runtime_deadline" if clock.hit else "eos" if final in ids else "max_tokens" if n>=max_new else "stopped"
    total=finished-started
    return {"raw_output":raw,"generated_tokens":n,"prompt_tokens":width,"finish_reason":reason,"stop_token_id":final if final in ids else None,"configured_eos_token_ids":sorted(ids),"total_latency_seconds":total,"ttft_seconds":(clock.first-started if clock.first else total),"tokens_per_second":n/total if total else 0.0,"peak_vram_bytes":int(torch.cuda.max_memory_allocated())}

def prompt(pack:dict[str,Any],case:dict[str,Any])->str:
    evidence="\n".join(f"- {x['ref']}: {x['fact']}" for x in case["evidence"])
    allowed=", ".join(case["allowed_evidence_refs"])
    keys=", ".join(pack["response_schema"]["required_exact_keys"])
    return f"""You are acting only as the Hephaestus DIAGNOSIS role.
ROLE BOUNDARY:
{pack['role_rule']}
SITUATION:
{case['situation']}
EVIDENCE:
{evidence}
Return exactly one JSON object and nothing else. Do not use markdown or code fences.
The object must contain exactly these keys: {keys}.
- decision: string
- action: string
- primary_variable: string
- confidence: JSON number from 0 to 1
- evidence_refs: JSON array. Cite only material evidence; relevant refs are a subset of: {allowed}
- uncertainties: JSON array of short strings; use [] only when no material uncertainty remains
- rationale: one concise string; do not invent evidence
Do not reveal hidden reasoning. Give only the requested decision record."""

def summary(rows:list[dict[str,Any]],cfg:dict[str,Any])->dict[str,Any]:
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    q=mean([r["score"]["quality_100"] for r in rows]); schema=mean([float(r["score"]["schema_compliant"]) for r in rows])
    ground=mean([r["score"]["components"]["evidence_grounding"] for r in rows]); cal=mean([r["score"]["components"]["confidence_calibration"] for r in rows])
    hall=mean([r["score"]["hallucination_rate"] for r in rows]); d1=[r for r in rows if r["root_case_id"]=="D1"]; d1q=mean([r["score"]["quality_100"] for r in d1])
    g=cfg["zero_shot_gate"]
    viable=(len(rows)==30 and q>=g["minimum_overall_quality_100"] and schema>=g["minimum_schema_compliance"] and ground>=g["minimum_evidence_grounding"] and hall<=g["maximum_hallucination_rate"] and d1q>=g["minimum_causal_restraint_quality_100"])
    return {"sample_count":len(rows),"quality_100":q,"schema_compliance":schema,"evidence_grounding":ground,"confidence_calibration":cal,"hallucination_rate":hall,"causal_restraint_quality_100":d1q,"exact_contract_pass_rate":mean([float(r["score"]["quality_100"]>=99.999) for r in rows]),"mean_generated_tokens":mean([r["generation"]["generated_tokens"] for r in rows]),"mean_latency_seconds":mean([r["generation"]["total_latency_seconds"] for r in rows]),"peak_vram_bytes":max([r["generation"]["peak_vram_bytes"] for r in rows],default=0),"viable_for_matched_micro_lora":viable}

def main()->int:
    cfg=json.loads(CONFIG_PATH.read_text()); pack=pack_from_builder(cfg); pack_sha=canonical_sha(pack)
    run_id=req("HEPHAESTUS_DIAG_FOUNDATION_RUN_ID"); repo_sha=req("HEPHAESTUS_REPO_SHA"); client=base.s3_client()
    prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"; deadline=time.monotonic()+int(cfg["execution"]["hard_wall_seconds"]); seed=int(cfg["generation"]["seed"])
    result={"result_version":"hephaestus-diagnosis-foundation-bakeoff.v1","status":"running","run_id":run_id,"repo_sha":repo_sha,"pack_sha256":pack_sha,"training_performed":False,"candidate_summaries":{},"started_at_unix":time.time()}
    completed=0
    def heartbeat(stage:str,**extra):
        p={"run_id":run_id,"stage":stage,"completed_samples":completed,"expected_samples":30*len(cfg["candidates"]),"timestamp_unix":time.time(),"remaining_wall_seconds":max(0,deadline-time.monotonic()),**extra}
        put(client,f"{prefix}/progress.json",p);print("DIAG_FOUNDATION_HEARTBEAT_JSON "+json.dumps(p,sort_keys=True),flush=True)
    put(client,f"{prefix}/run_manifest.json",{"protocol_id":cfg["protocol_id"],"repo_sha":repo_sha,"pack_sha256":pack_sha,"candidates":cfg["candidates"],"generation":cfg["generation"],"zero_shot_gate":cfg["zero_shot_gate"],"training_performed":False})
    cases=pack["partitions"]["zero_shot"]
    try:
        for ci,cand in enumerate(cfg["candidates"],1):
            if time.monotonic()>=deadline: raise TimeoutError("bakeoff hard wall before candidate")
            slug=cand["candidate_id"]; root=Path("/opt/hephaestus-diagnosis-foundation")/run_id/slug
            heartbeat("materializing_model",candidate_id=slug,candidate_index=ci)
            snap,adapter,manifest=materialize(cand,root/"model");put(client,f"{prefix}/models/{slug}/model_manifest.json",manifest)
            heartbeat("loading_model",candidate_id=slug,candidate_index=ci)
            model,tokenizer,runtime=load(cand,snap,adapter,cfg);put(client,f"{prefix}/models/{slug}/runtime.json",runtime)
            rows=[]
            heartbeat("model_loaded",candidate_id=slug,candidate_index=ci,allocated_memory_bytes=runtime["allocated_memory_bytes_after_load"])
            for ix,case in enumerate(cases,1):
                if time.monotonic()>=deadline: raise TimeoutError("bakeoff hard wall during probes")
                heartbeat("probe_started",candidate_id=slug,case_id=case["case_id"],case_index=ix)
                gen=generate(model,tokenizer,cand,prompt(pack,case),seed,int(cfg["generation"]["max_new_tokens"]),deadline)
                out=project(gen.pop("raw_output"),cand)
                score=topo.score_response(pack,case,out)
                row={"sample_version":"diagnosis-foundation-zero-shot.v1","candidate_id":slug,"model_id":cand["model_id"],"revision":cand["revision"],"case_id":case["case_id"],"root_case_id":case["root_case_id"],"condition":case["condition"],"seed":seed,"output":out,"generation":gen,"score":score}
                rows.append(row);completed+=1;put(client,f"{prefix}/samples/{slug}/{ix:03d}-{case['case_id']}.json",row)
                heartbeat("probe_complete",candidate_id=slug,case_id=case["case_id"],case_index=ix,quality_100=score["quality_100"],schema_compliant=score["schema_compliant"])
            sm=summary(rows,cfg);result["candidate_summaries"][slug]=sm;put(client,f"{prefix}/models/{slug}/summary.json",sm)
            del model,tokenizer;gc.collect()
            import torch
            torch.cuda.empty_cache();shutil.rmtree(root,ignore_errors=True)
            heartbeat("candidate_complete",candidate_id=slug,candidate_index=ci,summary=sm)
        eligible=[c["candidate_id"] for c in cfg["candidates"] if c.get("adaptation_candidate") and result["candidate_summaries"].get(c["candidate_id"],{}).get("viable_for_matched_micro_lora")]
        result.update({"status":"completed","completed_at_unix":time.time(),"completed_samples":completed,"expected_samples":30*len(cfg["candidates"]),"stage2_eligible_candidate_ids":eligible,"stage2_allowed_by_this_result":bool(eligible),"disposition":"zero_shot_complete_with_viable_foundation" if eligible else "zero_shot_complete_no_viable_foundation"})
        put(client,f"{prefix}/result.json",result);heartbeat("complete",stage2_eligible_candidate_ids=eligible)
        print("DIAG_FOUNDATION_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);return 0
    except Exception as exc:
        result.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"completed_at_unix":time.time(),"completed_samples":completed})
        put(client,f"{prefix}/result.json",result);print("DIAG_FOUNDATION_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);raise

if __name__=="__main__": raise SystemExit(main())
