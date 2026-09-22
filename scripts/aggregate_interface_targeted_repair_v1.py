#!/usr/bin/env python3
"""Aggregate ordered targeted repair role results without promoting them."""
from __future__ import annotations
import json,os
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/experiments/hephaestus_interface_targeted_repair_v1.json"

def req(name):
    v=(os.environ.get(name) or "").strip()
    if not v:raise RuntimeError("missing "+name)
    return v

def client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3",endpoint_url=req("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),region_name=req("RUNPOD_DATACENTER_ID"),aws_access_key_id=req("RUNPOD_S3_ACCESS_KEY_ID"),aws_secret_access_key=req("RUNPOD_S3_SECRET_ACCESS_KEY"),config=Config(retries={"mode":"standard","max_attempts":10}))

def get(c,b,k):
    r=c.get_object(Bucket=b,Key=k)
    try:v=json.loads(r["Body"].read().decode())
    finally:r["Body"].close()
    if not isinstance(v,dict):raise RuntimeError(k+" not object")
    return v

def put(c,b,k,v):
    raw=(json.dumps(v,indent=2,sort_keys=True)+"\n").encode();c.put_object(Bucket=b,Key=k,Body=raw)

def main():
    cfg=json.loads(CFG_PATH.read_text());run_id=req("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID");c=client();b=req("RUNPOD_NETWORK_VOLUME_ID")
    prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}";roles={}
    for role in cfg["training"]["roles_in_order"]:
        row=get(c,b,f"{prefix}/roles/{role}/result.json")
        if row.get("status")!="completed" or row.get("selected_dev_gate",{}).get("passed") is not True:
            raise RuntimeError(f"{role} repair not passing")
        roles[role]={
          "selected_step":row["selected_step"],"selected_dev_summary":row["selected_dev_summary"],
          "selected_adapter":row["selected_adapter"],"training_presentation":row["training_presentation"],
          "source_adapter_sha256":get(c,b,f"{prefix}/roles/{role}/model.json")["source_adapter_sha256"],
        }
    presentations={x["training_presentation"] for x in roles.values()}
    if len(presentations)!=1:raise RuntimeError("repair roles used inconsistent handoff presentations")
    result={"result_version":"hephaestus-interface-repair-aggregate.v1","status":"completed","run_id":run_id,
      "roles":roles,"training_presentation":next(iter(presentations)),"ready_for_phase_ii_b_stack_assembly":True,
      "production_promotion_performed":False,"automatic_role_dispatch_enabled":False}
    put(c,b,f"{prefix}/aggregate.json",result)
    print("INTERFACE_REPAIR_AGGREGATE_JSON "+json.dumps(result,sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
