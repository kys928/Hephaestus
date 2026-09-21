#!/usr/bin/env python3
from __future__ import annotations
import json,os
from pathlib import Path
import launch_first_bounded_scientific_training as storage
import role_mastery_common_v1 as common
def main():
 cfg=common.load_cfg();gid=os.environ["GITHUB_RUN_ID"];client=storage.s3_client();roles=sorted(cfg["candidate_registry"]);results={}
 for role in roles:
  run_id=f"role-mastery-phase1-{gid}-{role}";key=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}/result.json"
  raw=storage.read_key(client,key);results[role]=json.loads(raw)
 aggregate={"result_version":"hephaestus-role-mastery-phase1-aggregate.v1","workflow_run_id":gid,"roles":{r:{"status":v["status"],"model_id":v["candidate"]["model_id"],"selected_epoch":v.get("selected_epoch"),"certification_passed":v.get("certification_gate",{}).get("passed"),"cert_quality_100":v.get("certification",{}).get("quality_100"),"selected_adapter":v.get("selected_adapter")} for r,v in results.items()}}
 aggregate["all_roles_completed"]=all(v["status"]=="completed" for v in results.values())
 aggregate["all_roles_certified_for_phase2"]=all(bool(v.get("certification_gate",{}).get("passed")) for v in results.values())
 aggregate["status"]="completed" if aggregate["all_roles_completed"] else "incomplete"
 key=f"{cfg['execution']['s3_prefix'].rstrip('/')}/role-mastery-phase1-{gid}/aggregate_result.json"
 client.put_object(Bucket=storage.VOLUME_ID,Key=key,Body=(json.dumps(aggregate,indent=2,sort_keys=True)+"\n").encode())
 print("ROLE_MASTERY_AGGREGATE_JSON "+json.dumps(aggregate,sort_keys=True))
if __name__=="__main__":main()
