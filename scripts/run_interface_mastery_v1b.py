#!/usr/bin/env python3
"""Run fresh adversarial Phase II-B against the repaired role stack.

Diagnosis and Planner remain the certified Phase I adapters. Controller, Evaluator,
and Judge are overlaid from the completed targeted-repair aggregate. The holdout is
read only here, after training has fully completed.
"""
from __future__ import annotations

import dataclasses
import gc
import importlib.util
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from hephaestus.evaluation.handoff_normalization import render_normalized_handoff
from hephaestus.evaluation.interface_mastery import score_interface_case, summarize_scorecards
from hephaestus.providers.models.role_stack import AdapterArtifact, load_certified_role_model_stack

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/experiments/hephaestus_interface_mastery_v1b.json"


def load_module(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_json(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text())
    if not isinstance(value,dict): raise RuntimeError(f"{path} must be object")
    return value


def get_json(client:Any,bucket:str,key:str)->dict[str,Any]:
    response=client.get_object(Bucket=bucket,Key=key)
    try:value=json.loads(response["Body"].read().decode())
    finally:response["Body"].close()
    if not isinstance(value,dict):raise RuntimeError(f"{key} is not object")
    return value


def put_json(client:Any,bucket:str,key:str,payload:object)->None:
    raw=(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    client.put_object(Bucket=bucket,Key=key,Body=raw)
    response=client.get_object(Bucket=bucket,Key=key)
    try:observed=response["Body"].read()
    finally:response["Body"].close()
    if observed!=raw:raise RuntimeError(f"S3 readback mismatch {key}")


class RepairedStack:
    def __init__(self,base,roles):
        self.base=base;self.roles=roles
        self.automatic_role_dispatch_enabled=False
    def resolve(self,role):
        return self.roles[role]


def repaired_stack(cfg:dict[str,Any],client:Any,bucket:str):
    base=load_certified_role_model_stack(ROOT/cfg["source_stack"]["registry_path"])
    if base.source_run_id!=cfg["source_stack"]["required_source_run_id"]:
        raise RuntimeError("Phase II-B source stack mismatch")
    repair_run=str(cfg["repair_gate"].get("required_repair_run_id") or "")
    if not repair_run:raise RuntimeError("Phase II-B repair run is not frozen")
    key=f"{cfg['repair_gate']['s3_prefix'].rstrip('/')}/{repair_run}/aggregate.json"
    aggregate=get_json(client,bucket,key)
    if aggregate.get("status")!="completed" or aggregate.get("ready_for_phase_ii_b_stack_assembly") is not True:
        raise RuntimeError("targeted repair aggregate is not ready for Phase II-B")
    roles=dict(base.roles)
    repaired=set(cfg["repair_gate"]["required_roles"])
    if set(aggregate.get("roles",{}))!=repaired:
        raise RuntimeError("repair aggregate role coverage mismatch")
    for role in repaired:
        row=aggregate["roles"][role]
        adapter=row["selected_adapter"]
        roles[role]=dataclasses.replace(
            roles[role],
            adapter=AdapterArtifact(
                s3_key=str(adapter["s3_key"]),
                sha256=str(adapter["sha256"]),
                bytes=int(adapter["bytes"]),
            ),
        )
    return RepairedStack(base,roles),aggregate


def main()->int:
    cfg=load_json(CFG_PATH)
    if cfg["governance"]["training_use_allowed"] or cfg["governance"]["weights_mutation_allowed"]:
        raise RuntimeError("Phase II-B must remain evaluation-only")
    iib=load_module(ROOT/cfg["pack"]["builder_path"],"phase_iib_pack")
    pack=iib.build_pack();iib.validate(pack)
    observed=iib.canonical_sha256(pack)
    if observed!=cfg["pack"]["canonical_sha256"]:
        raise RuntimeError(f"Phase II-B pack drift {observed}")

    v1=load_module(ROOT/"scripts/run_interface_mastery_v1.py","phase_iib_v1_runtime")
    source_cfg=load_json(ROOT/"configs/experiments/hephaestus_interface_mastery_v1.json")
    run_id=v1.required("HEPHAESTUS_INTERFACE_MASTERY_V1B_RUN_ID")
    repo_sha=v1.required("HEPHAESTUS_REPO_SHA")
    client=v1.s3_client();bucket=v1.bucket()
    stack,repair_aggregate=repaired_stack(cfg,client,bucket)
    presentation=str(repair_aggregate.get("training_presentation",""))
    if presentation not in {"raw_handoff","normalized_handoff"}:
        raise RuntimeError("repaired stack has no valid handoff presentation")

    runtime_cfg=dict(source_cfg)
    runtime_cfg["execution"]=dict(cfg["execution"])
    runtime_cfg["runtime"]=source_cfg["runtime"]
    prefix=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    root=Path(f"/tmp/{run_id}");shutil.rmtree(root,ignore_errors=True);root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+float(cfg["execution"]["hard_wall_seconds"])-60
    result={
      "result_version":"hephaestus-interface-mastery-v1b.v1","protocol_id":cfg["protocol_id"],
      "run_id":run_id,"repo_sha":repo_sha,"pack_sha256":observed,
      "repair_run_id":cfg["repair_gate"]["required_repair_run_id"],
      "handoff_presentation":presentation,"status":"running","sample_count":0,
      "weights_mutated":False,"production_promotion_performed":False,
      "automatic_role_dispatch_enabled":False,"started_at_unix":time.time(),
    }
    scores=[]
    seed=int(cfg["generation"]["seed"])
    try:
      for interface_index,interface_id in enumerate(cfg["interfaces"],1):
        cases=pack["partitions"][interface_id];producer_role,consumer_role=interface_id.split("_to_")
        producer_model,producer_tokenizer,producer_runtime=v1.load_role_model(producer_role,stack,runtime_cfg,client,root/"assets")
        upstream={}
        for case_index,case in enumerate(cases,1):
          generation=v1.generate(producer_model,producer_tokenizer,source_cfg["runtime"][producer_role],v1.prompt_for_role(pack,case,producer_role),seed+interface_index*100+case_index,deadline)
          upstream[case["case_id"]]=generation
        del producer_model,producer_tokenizer;v1.release_cuda();gc.collect()

        consumer_model,consumer_tokenizer,consumer_runtime=v1.load_role_model(consumer_role,stack,runtime_cfg,client,root/"assets")
        for case_index,case in enumerate(cases,1):
          producer_generation=upstream[case["case_id"]];raw_upstream=str(producer_generation["projected_output"])
          handoff=raw_upstream if presentation=="raw_handoff" else render_normalized_handoff(source_role=producer_role,raw_output=raw_upstream)
          consumer_generation=v1.generate(
            consumer_model,consumer_tokenizer,source_cfg["runtime"][consumer_role],
            v1.prompt_for_role(pack,case,consumer_role,handoff),seed+10000+interface_index*100+case_index,deadline,
          )
          boundary_pass=True;boundary=None
          if interface_id=="judge_to_controller":
            boundary_pass,boundary=v1.controller_boundary(case,str(consumer_generation["projected_output"]),run_id)
          score=score_interface_case(
            case=case,producer_raw=raw_upstream,consumer_raw=str(consumer_generation["projected_output"]),
            producer_vocabulary=pack["contract_vocabulary"][producer_role],
            consumer_vocabulary=pack["contract_vocabulary"][consumer_role],
            deterministic_boundary_passed=boundary_pass,
          )
          scores.append(score)
          put_json(client,bucket,f"{prefix}/samples/{interface_id}/{case_index:02d}-{case['case_id']}.json",{
            "case":case,"producer_generation":producer_generation,"handoff_presentation":presentation,
            "consumer_generation":consumer_generation,"deterministic_boundary":boundary,"scorecard":score.to_dict(),
          })
        del consumer_model,consumer_tokenizer;v1.release_cuda();gc.collect()

      summary=summarize_scorecards(scores,cfg["certification"],protocol_id=cfg["protocol_id"])
      result.update({"status":"completed","sample_count":len(scores),"summary":summary,"completed_at_unix":time.time()})
      put_json(client,bucket,f"{prefix}/summary.json",summary);put_json(client,bucket,f"{prefix}/result.json",result)
      print("INTERFACE_MASTERY_V1B_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True)
      return 0
    except Exception as exc:
      result.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"completed_at_unix":time.time()})
      put_json(client,bucket,f"{prefix}/result.json",result)
      print("INTERFACE_MASTERY_V1B_RESULT_JSON "+json.dumps(result,sort_keys=True),flush=True)
      raise
    finally:
      shutil.rmtree(root,ignore_errors=True)


if __name__=="__main__":raise SystemExit(main())
