#!/usr/bin/env python3
"""Read-only CUDA inference test of one actual pinned original Hephaestus role.

Loads the CONTROLLER Qwen3-4B base at its exact 40-char HF revision, applies
its original SHA-verified LoRA adapter from Global Volume, and generates 8 tokens.
Never starts training, modifies registered weights, or promotes a checkpoint.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, sys, tarfile, time
from datetime import datetime,timezone
import boto3

GLOBAL=pathlib.Path("/workspace-global")
GLOBAL_ID="cmw51pjsk000007l9a0mv1jga"
SOURCE_ID="cviwpryzao"
BASE="Qwen/Qwen3-4B-Instruct-2507"
BASE_REV="cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_SHA="62ae757aad46191486c09acf87e164972434b19834f4401f9dd7f3e8bb65972e"
ADAPTER_BYTES=43618367
ADAPTER=GLOBAL/"hephaestus/runtime-models/role-mastery-v1/controller"/("selected-adapter-"+ADAPTER_SHA[:24]+".tar.gz")
HISTORICAL=GLOBAL/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
BUF=4*1024*1024
def hash_file(path):
 h=hashlib.sha256();count=0
 with path.open("rb") as f:
  while block:=f.read(BUF):h.update(block);count+=len(block)
 return h.hexdigest(),count

def main():
 run=os.environ.get("GPU_SMOKE_RUN_ID","")
 if not run.isdecimal():raise RuntimeError("Run identity absent")
 if os.environ.get("RUNPOD_GLOBAL_VOLUME_ID")!=GLOBAL_ID or not GLOBAL.is_mount() or not HISTORICAL.is_file():
  raise RuntimeError("Wrong/unmounted Global Volume")
 token=os.environ.get("HF_TOKEN")
 if not token:raise RuntimeError("Hugging Face token absent")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
  aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 def phase(state,details=None):
  record={"schema":"hephaestus.original-role-cuda-smoke.v1","run_id":run,
   "status":state,"role":"controller","base_model":BASE,"base_revision":BASE_REV,
   "adapter_sha256":ADAPTER_SHA,"global_volume_id":GLOBAL_ID,
   "updated_at":datetime.now(timezone.utc).isoformat()}
  if details:record.update(details)
  s3.put_object(Bucket=SOURCE_ID,Key="hephaestus/gpu-smoke-status/"+run+".json",
   Body=json.dumps(record,sort_keys=True).encode(),ContentType="application/json")
  print("GPU_MODEL_SMOKE_PHASE="+state,flush=True)
 phase("STARTED")
 if ADAPTER.is_symlink() or not ADAPTER.is_file() or hash_file(ADAPTER)!=(ADAPTER_SHA,ADAPTER_BYTES):
  raise RuntimeError("Pinned trained Controller adapter missing or different on Global")
 phase("GLOBAL_ADAPTER_HASH_VERIFIED")
 archive_dir=pathlib.Path("/tmp/hephaestus-adapter")
 archive_dir.mkdir(exist_ok=True)
 with tarfile.open(ADAPTER,"r:gz") as tf:
  seen=set();total=0
  for item in tf.getmembers():
   p=pathlib.PurePosixPath(item.name)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!="adapter":
    raise RuntimeError("Unsafe PEFT archive structure")
   if item.issym() or item.islnk() or item.isdev():raise RuntimeError("Symlink/device in PEFT archive")
   if item.isdir():continue
   if not item.isfile() or len(p.parts)!=2 or p.name in seen:raise RuntimeError("Unexpected adapter member")
   total+=item.size
   if total>900*1024*1024:raise RuntimeError("Adapter unpack exceeds 900 MB")
   src=tf.extractfile(item)
   if src is None:raise RuntimeError("Unreadable adapter member")
   with src,(archive_dir/p.name).open("xb") as out:
    while chunk:=src.read(BUF):out.write(chunk)
   seen.add(p.name)
 if not {"adapter_config.json","adapter_model.safetensors"}.issubset(seen):
  raise RuntimeError("Incomplete LoRA adapter")
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
  raise RuntimeError("One real NVIDIA CUDA GPU required")
 gpu=torch.cuda.get_device_name(0)
 print("GPU_HARDWARE="+gpu,flush=True)
 phase("PINNED_BASE_LOADING",{"gpu":gpu})
 tokenizer=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV,token=token,cache_dir="/tmp/hephaestus-base-hf",
  trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(BASE,revision=BASE_REV,token=token,
  cache_dir="/tmp/hephaestus-base-hf",dtype=torch.bfloat16,device_map="cuda:0",
  low_cpu_mem_usage=True,trust_remote_code=False)
 model=PeftModel.from_pretrained(model,str(archive_dir),is_trainable=False)
 model.eval()
 phase("PINNED_BASE_AND_PEFT_ADAPTER_LOADED",{"gpu":gpu})
 prompt="Reply with exactly one word: READY"
 tokens=tokenizer(prompt,return_tensors="pt")
 tokens={k:v.to("cuda:0") for k,v in tokens.items()}
 with torch.inference_mode():
  generated=model.generate(**tokens,max_new_tokens=8,do_sample=False,pad_token_id=tokenizer.eos_token_id)
 if generated.shape[1]<=tokens["input_ids"].shape[1]:
  raise RuntimeError("CUDA inference produced no new tokens")
 out=generated[0].detach().cpu().tolist()
 digest=hashlib.sha256(json.dumps(out,separators=(",",":")).encode()).hexdigest()
 phase("COMPLETE",{"gpu":gpu,"generated_tokens":int(generated.shape[1]-tokens["input_ids"].shape[1]),
  "token_ids_sha256":digest,"base_and_lora_loaded_on_cuda":True})
 print("REAL_ORIGINAL_CONTROLLER_CUDA_INFERENCE_PASSED=true",flush=True)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("REAL_CUDA_ROLE_SMOKE_FAILED="+type(exc).__name__+": "+str(exc)[:140],flush=True)
  try:
   run=os.environ.get("GPU_SMOKE_RUN_ID","")
   if run.isdecimal():
    s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
     aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
     aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
    s3.put_object(Bucket=SOURCE_ID,Key="hephaestus/gpu-smoke-status/"+run+".json",
      Body=json.dumps({"schema":"hephaestus.original-role-cuda-smoke.v1","run_id":run,
       "status":"FAILED","error_class":type(exc).__name__,"error_summary":str(exc)[:120]}).encode(),
      ContentType="application/json")
  except Exception:print("GPU_SMOKE_FAILURE_REPORT_UNAVAILABLE=true",flush=True)
  sys.exit(1)
