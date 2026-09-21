#!/usr/bin/env python3
from __future__ import annotations
import gc,hashlib,json,os,random,shutil,tarfile,time,traceback
from pathlib import Path
from typing import Any
SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in os.sys.path: os.sys.path.insert(0,str(SCRIPTS))
import role_mastery_common_v1 as common
import run_cognitive_topology_v1 as topo
import run_diagnosis_foundation_bakeoff_v1 as modelio
import launch_first_bounded_scientific_training as storage
ROOT=Path(__file__).resolve().parents[1]

def req(k:str)->str:
 v=(os.environ.get(k) or "").strip()
 if not v: raise RuntimeError("missing "+k)
 return v
def put(client:Any,key:str,obj:object)->None:
 raw=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
 client.put_object(Bucket=storage.VOLUME_ID,Key=key,Body=raw)
def sha_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for ch in iter(lambda:f.read(1024*1024),b""): h.update(ch)
 return h.hexdigest()
def save_adapter(model:Any,root:Path,client:Any,key:str)->dict[str,Any]:
 root.mkdir(parents=True,exist_ok=True);model.save_pretrained(root,safe_serialization=True)
 tar=root.parent/(root.name+".tar.gz")
 with tarfile.open(tar,"w:gz") as tf: tf.add(root,arcname="adapter")
 client.upload_file(str(tar),storage.VOLUME_ID,key)
 return {"s3_key":key,"sha256":sha_file(tar),"bytes":tar.stat().st_size,"local_dir":str(root)}
def evaluate(model:Any,tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],cases:list[dict[str,Any]],seeds:list[int],max_new:int,deadline:float,client:Any,prefix:str,phase:str,heartbeat)->tuple[list[dict[str,Any]],dict[str,Any]]:
 rows=[]
 for seed in seeds:
  for ix,case in enumerate(cases,1):
   if time.monotonic()>=deadline: raise TimeoutError("role mastery hard wall during evaluation")
   gen=modelio.generate(model,tokenizer,cand,common.prompt(pack,case),seed,max_new,deadline)
   raw=gen.pop("raw_output")
   normalized,extract=modelio.extract_complete_json(raw)
   score=topo.score_response(pack,case,normalized)
   row={"sample_version":"role-mastery-phase1.v1","role":case["role"],"phase":phase,"seed":seed,"case_id":case["case_id"],"root_case_id":case["root_case_id"],"skill":case["skill"],"raw_output":raw,"output":normalized,"extraction":extract,"generation":gen,"score":score}
   rows.append(row);put(client,f"{prefix}/samples/{phase}/seed-{seed}/{ix:03d}-{case['case_id']}.json",row)
  heartbeat("evaluation_seed_complete",phase=phase,seed=seed,quality_100=common.summary([r for r in rows if r["seed"]==seed])["quality_100"])
 return rows,common.summary(rows)
def set_train_mode(model:Any,enabled:bool)->None:
 if enabled:
  model.train()
  if hasattr(model,"gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
  if hasattr(model.config,"use_cache"): model.config.use_cache=False
 else:
  model.eval()
  if hasattr(model,"gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
  if hasattr(model.config,"use_cache"): model.config.use_cache=True
def train_epoch(model:Any,examples:list[dict[str,Any]],optimizer:Any,cfg:dict[str,Any],epoch:int,deadline:float,heartbeat,role:str)->dict[str,Any]:
 import torch
 accum=int(cfg["training"]["gradient_accumulation_steps"]);seed=int(cfg["training"]["seed"])+epoch
 order=list(range(len(examples)));random.Random(seed).shuffle(order)
 if len(order)%accum: raise RuntimeError("training case count must divide gradient accumulation")
 losses=[];sup=nonpad=slots=0;cursor=0
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
 set_train_mode(model,True)
 for step in range(1,len(order)//accum+1):
  if time.monotonic()>=deadline: raise TimeoutError("role mastery hard wall during training")
  optimizer.zero_grad(set_to_none=True);sl=0.0
  for _ in range(accum):
   ex=examples[order[cursor]];cursor+=1
   batch={k:v.to("cuda",non_blocking=True) for k,v in ex.items() if k in {"input_ids","attention_mask","labels"}}
   out=model(**batch,use_cache=False);loss=out.loss/accum
   if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss epoch={epoch} step={step}")
   loss.backward();sl+=float(loss.detach().cpu())*accum
   sup+=int(ex["supervised_tokens"]);nonpad+=int(ex["nonpad_tokens"]);slots+=int(cfg["training"]["max_sequence_length"])
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg["training"]["max_grad_norm"]))
  optimizer.step();losses.append(sl/accum)
  if step==1 or step%25==0 or step==len(order)//accum:
   heartbeat("training_step",role=role,epoch=epoch,optimizer_step=step,optimizer_steps=len(order)//accum,loss=losses[-1])
 torch.cuda.synchronize();seconds=time.perf_counter()-started
 set_train_mode(model,False)
 return {"epoch":epoch,"optimizer_steps":len(losses),"training_seconds":seconds,"loss_first":losses[0],"loss_last":losses[-1],"loss_mean":sum(losses)/len(losses),"supervised_token_updates":sup,"nonpad_token_updates":nonpad,"padded_training_token_slots":slots,"peak_training_vram_bytes":int(torch.cuda.max_memory_allocated())}
def choose_epoch(records:list[dict[str,Any]],cfg:dict[str,Any])->dict[str,Any]:
 def key(r):
  g=common.hard_gates(r["dev"],cfg)
  return (int(all(g.values())),r["dev"]["quality_100"],r["dev"]["exact_contract_pass_rate"],r["dev"]["evidence_grounding"],-r["dev"]["hallucination_rate"])
 return max(records,key=key)
def main()->int:
 import torch
 from peft import LoraConfig,PeftModel,get_peft_model
 cfg=common.load_cfg();builder,pack=common.load_pack(cfg)
 if builder.canonical_sha256(pack)!=cfg["pack"]["canonical_sha256"]: raise RuntimeError("pack hash drift")
 role=req("HEPHAESTUS_ROLE_MASTERY_ROLE");run_id=req("HEPHAESTUS_ROLE_MASTERY_RUN_ID");repo_sha=req("HEPHAESTUS_REPO_SHA")
 if role not in cfg["candidate_registry"]: raise RuntimeError("unknown role "+role)
 cand=dict(cfg["candidate_registry"][role]);cand["decoding"]={"do_sample":False}
 client=storage.s3_client();prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}";deadline=time.monotonic()+int(cfg["execution"]["hard_wall_seconds"])
 result={"result_version":"hephaestus-role-mastery-phase1.v1","status":"running","role":role,"run_id":run_id,"repo_sha":repo_sha,"pack_sha256":cfg["pack"]["canonical_sha256"],"candidate":cand,"started_at_unix":time.time(),"production_promotion_performed":False}
 def heartbeat(stage:str,**extra):
  p={"run_id":run_id,"role":role,"stage":stage,"timestamp_unix":time.time(),"remaining_wall_seconds":max(0,deadline-time.monotonic()),**extra}
  put(client,f"{prefix}/progress.json",p);print("ROLE_MASTERY_HEARTBEAT_JSON "+json.dumps(p,sort_keys=True),flush=True)
 put(client,f"{prefix}/run_manifest.json",{"protocol":cfg,"role":role,"candidate":cand,"repo_sha":repo_sha,"pack_sha256":cfg["pack"]["canonical_sha256"]})
 root=Path("/opt/hephaestus-role-mastery")/run_id
 try:
  heartbeat("materializing_model")
  snap,adapter,manifest=modelio.materialize(cand,root/"model");put(client,f"{prefix}/model_manifest.json",manifest)
  model,tokenizer,runtime=modelio.load(cand,snap,adapter,cfg);put(client,f"{prefix}/runtime.json",runtime)
  targets=common.target_modules(model,cand,list(cfg["training"]["target_module_suffixes"]))
  lc=LoraConfig(r=int(cfg["training"]["lora_rank"]),lora_alpha=int(cfg["training"]["lora_alpha"]),lora_dropout=float(cfg["training"]["lora_dropout"]),bias="none",target_modules=targets)
  model=get_peft_model(model,lc)
  for n,p in model.named_parameters():
   if "lora_" not in n: p.requires_grad_(False)
  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
  if trainable<=0: raise RuntimeError("zero trainable LoRA parameters")
  if role in {"diagnosis","planner"} and any("vision_tower" in n for n,p in model.named_parameters() if p.requires_grad): raise RuntimeError("vision tower became trainable")
  maxlen=int(cfg["training"]["max_sequence_length"]);train_cases=pack["partitions"][role]["train"]
  examples=[common.training_example(tokenizer,cand,pack,c,maxlen) for c in train_cases]
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]))
  epochs=[]
  for epoch in cfg["training"]["epochs"]:
   heartbeat("epoch_started",epoch=epoch,trainable_parameters=trainable,target_module_count=len(targets))
   tr=train_epoch(model,examples,optimizer,cfg,int(epoch),deadline,heartbeat,role)
   art=save_adapter(model,root/f"adapter-epoch-{epoch}",client,f"{prefix}/adapters/epoch-{epoch}.tar.gz")
   rows,dev=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["dev"],[int(cfg["evaluation"]["dev_seed"])],int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,prefix,f"dev-epoch-{epoch}",heartbeat)
   rec={"epoch":epoch,"training":tr,"dev":dev,"dev_hard_gates":common.hard_gates(dev,cfg),"adapter":art};epochs.append(rec);put(client,f"{prefix}/epochs/epoch-{epoch}.json",rec)
   heartbeat("epoch_complete",epoch=epoch,dev_quality_100=dev["quality_100"],dev_hard_gates=rec["dev_hard_gates"])
  selected=choose_epoch(epochs,cfg);selected_epoch=int(selected["epoch"]);selected_dir=Path(selected["adapter"]["local_dir"])
  heartbeat("epoch_selected",selected_epoch=selected_epoch,dev_quality_100=selected["dev"]["quality_100"])
  del model;gc.collect();torch.cuda.empty_cache()
  base_model,tokenizer,runtime2=modelio.load(cand,snap,None,cfg)
  model=PeftModel.from_pretrained(base_model,str(selected_dir),is_trainable=False);model.eval()
  cert_rows,cert=evaluate(model,tokenizer,cand,pack,pack["partitions"][role]["cert"],[int(x) for x in cfg["evaluation"]["certification_seeds"]],int(cfg["evaluation"]["max_new_tokens_by_role"][role]),deadline,client,prefix,"certification",heartbeat)
  certification=common.certification(cert,cfg)
  final_art={k:v for k,v in selected["adapter"].items() if k!="local_dir"}
  result.update({"status":"completed","completed_at_unix":time.time(),"selected_epoch":selected_epoch,"epochs":epochs,"certification":cert,"certification_gate":certification,"selected_adapter":final_art,"trainable_parameter_count":trainable,"target_module_count":len(targets),"target_modules_sha256":hashlib.sha256("\n".join(targets).encode()).hexdigest(),"eligible_for_phase2":bool(certification["passed"])})
  put(client,f"{prefix}/result.json",result);heartbeat("complete",certification_passed=certification["passed"],eligible_for_phase2=result["eligible_for_phase2"],cert_quality_100=cert["quality_100"])
  print("ROLE_MASTERY_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);return 0
 except Exception as exc:
  result.update({"status":"failed","completed_at_unix":time.time(),"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
  put(client,f"{prefix}/result.json",result);print("ROLE_MASTERY_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True);raise
 finally:
  shutil.rmtree(root,ignore_errors=True)
if __name__=="__main__": raise SystemExit(main())
