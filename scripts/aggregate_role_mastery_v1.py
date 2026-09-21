#!/usr/bin/env python3
from __future__ import annotations
import json,os,time
from pathlib import Path
import sys
SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
import launch_first_bounded_scientific_training as storage
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/experiments/hephaestus_role_mastery_v1.json").read_text())
run_id=(os.environ.get("HEPHAESTUS_ROLE_MASTERY_RUN_ID") or "").strip()
if not run_id:raise RuntimeError("missing HEPHAESTUS_ROLE_MASTERY_RUN_ID")
client=storage.s3_client();base=f"{CFG['execution']['s3_prefix'].rstrip('/')}/{run_id}"
roles={}
for role in CFG["candidates"]:
    key=f"{base}/roles/{role}/result.json"
    try:roles[role]=json.loads(storage.read_key(client,key))
    except Exception as exc:roles[role]={"status":"missing","error":str(exc)}
all_completed=all(x.get("status")=="completed" for x in roles.values())
all_certified=all(bool(x.get("certification",{}).get("passed")) for x in roles.values() if x.get("status")=="completed") and all_completed
result={"result_version":"hephaestus-role-mastery-aggregate.v1","run_id":run_id,"status":"completed" if all_completed else "incomplete","all_roles_completed":all_completed,"all_roles_certified":all_certified,"roles":{r:{
 "status":x.get("status"),"model_id":x.get("model_id"),"selected_dev_step":x.get("selected_dev_step"),
 "certification":x.get("certification"),"certification_summary":x.get("certification_summary"),"selected_adapter":x.get("selected_adapter"),
 "error_type":x.get("error_type"),"error":x.get("error")} for r,x in roles.items()},"production_promotion_performed":False,"completed_at_unix":time.time()}
raw=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
client.put_object(Bucket=storage.VOLUME_ID,Key=f"{base}/result.json",Body=raw)
print("ROLE_MASTERY_AGGREGATE_JSON "+json.dumps(result,sort_keys=True))
if not all_completed:raise SystemExit(2)
