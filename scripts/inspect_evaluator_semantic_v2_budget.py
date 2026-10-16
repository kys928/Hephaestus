#!/usr/bin/env python3
"""Read-only control-plane/catalog/S3 inspection after a bounded launch refusal."""
import hashlib
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
from hephaestus.providers.runpod.execution import RunPodApiError
from launch_evaluator_semantic_repair_v2 import s3_client


def inspect():
    cfg=json.loads(Path('configs/experiments/hephaestus_evaluator_semantic_repair_v2.json').read_text())
    execution=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider())
    pod='lji9euultcursa'
    try:
        execution.get_pod(pod)
        absent=False
    except RunPodApiError as exc:
        absent=(isinstance(exc.__cause__,urllib.error.HTTPError) and exc.__cause__.code==404) or str(exc)=='unexpected get-Pod response status: 404'
        if not absent:raise
    query='query { gpuTypes(input: { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition" }) { id displayName memoryInGb lowestPrice(input: { gpuCount: 1, secureCloud: true }) { stockStatus uninterruptablePrice availableGpuCounts } } }'
    req=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':query}).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],
        'User-Agent':'Mozilla/5.0 (compatible; Hephaestus-RunPod/1.0)'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=20) as response:catalog=json.load(response)
    except urllib.error.HTTPError as exc:
        catalog={'request_failed_http_status':exc.code}
    s3=s3_client();bucket=os.environ['RUNPOD_NETWORK_VOLUME_ID']
    root=cfg['execution']['s3_prefix'];prefix=root+'/evaluator-semantic-v2-C-38933236763/'
    listing=s3.list_objects_v2(Bucket=bucket,Prefix=prefix,MaxKeys=100)
    objects=[dict(key=x['Key'],bytes=x['Size']) for x in listing.get('Contents',[])]
    claim=root+'/approvals/'+cfg['governance']['budget_approval_id']+'/train-claim.json'
    response=s3.get_object(Bucket=bucket,Key=claim)
    try:raw=response['Body'].read()
    finally:response['Body'].close()
    result=dict(failed_workflow_run=38933236763,pod_id=pod,pod_verified_absent=absent,
        current_catalog=catalog,approved_hourly_usd=cfg['execution']['max_hourly_usd_per_pod'],
        launch_claim_sha256=hashlib.sha256(raw).hexdigest(),launch_claim=json.loads(raw),run_objects=objects,
        actual_failed_pod_price_unrecorded=True,s3_writes=0,pod_allocations=0)
    Path('semantic-v2-budget-inspection.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('SEMANTIC_V2_BUDGET_INSPECTION '+json.dumps(result,sort_keys=True))


if __name__=='__main__':inspect()
