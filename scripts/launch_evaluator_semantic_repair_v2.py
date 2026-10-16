#!/usr/bin/env python3
"""Bounded C proof launcher. Dry-run by default; no paid GitHub workflow trigger."""
from __future__ import annotations
import argparse
import json
import os
import time
import launch_evaluator_semantic_invariance_repair_v1 as previous
import launch_interface_repair_v2 as infra
from prepare_evaluator_semantic_repair_v2 import prepare
from run_evaluator_semantic_repair_v2 import authorize
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter


def bounded_cost(snapshot, elapsed, cfg, mode):
    raw=snapshot.get('adjustedCostPerHr',snapshot.get('costPerHr'))
    if raw is None:
        raise ValueError('missing pod price; fail closed')
    import math
    hourly=float(raw)
    if not math.isfinite(hourly) or hourly<=0 or hourly>cfg['execution']['max_hourly_usd_per_pod']:
        raise ValueError(f'pod hourly price outside approved ceiling: observed={hourly!r}, approved={cfg["execution"]["max_hourly_usd_per_pod"]}')
    limit=cfg['execution']['max_estimated_total_usd_train' if mode=='train' else 'max_estimated_total_usd_evaluate']
    # Reserve six minutes for bounded API polling and five teardown attempts. The launcher and remote runner
    # enforce timeouts; verified deletion is required even after any exception.
    estimated=hourly*elapsed/3600
    if hourly*(elapsed+360)/3600>=limit:
        raise TimeoutError('pod run cost ceiling reached')
    return dict(cost_per_hr=hourly,estimated_cost_usd=estimated)


def request(cfg,repo_sha,run_id,mode,*,real_env):
    # Reuse the established bootstrap/dependency pins, replacing one exact body.
    body=previous.request_body(cfg,repo_sha,run_id,mode,96 if mode=='evaluate' else None,real_env=real_env)
    # RunPod's existing adapter uses dockerStartCmd or dockerArgs depending on
    # request normalization. Refuse to silently leave the old runner in place.
    text=json.dumps(body)
    old='scripts/run_evaluator_semantic_invariance_repair_v1.py'
    if old not in text:
        raise ValueError('established pod bootstrap runner contract changed')
    def replace_strings(value):
        if isinstance(value,str):
            value=value.replace(old,'scripts/run_evaluator_semantic_repair_v2.py')
            value=value.replace('export BOOTSTRAP_FILE READY HUGGINGFACE_HUB_CACHE',
                "export BOOTSTRAP_FILE READY HUGGINGFACE_HUB_CACHE\ntrap 'write_bootstrap bootstrap_failed' ERR")
            value=value.replace('--evaluate-checkpoint "$HEPHAESTUS_EVALUATOR_REPAIR_CHECKPOINT_STEP"','--mode evaluate --step 96 --arm C')
            value=value.replace('scripts/run_evaluator_semantic_repair_v2.py --train','scripts/run_evaluator_semantic_repair_v2.py --mode train --arm C')
            return value
        if isinstance(value,list):return [replace_strings(v) for v in value]
        if isinstance(value,dict):return {k:replace_strings(v) for k,v in value.items()}
        return value
    body=replace_strings(body)
    body['env']['HEPHAESTUS_EVALUATOR_V2_AUTHORIZED']='LAUNCH_APPROVED_EVALUATOR_SEMANTIC_V2'
    body['env']['HEPHAESTUS_EVALUATOR_V2_RUN_ID']=run_id
    body['name']=f'hephaestus-evaluator-semantic-v2-C-{mode}-{run_id}'[:180]
    return body


def delete_verified(execution,pod_id):
    from hephaestus.providers.runpod.execution import RunPodApiError
    import urllib.error
    errors=[]
    for attempt in range(1,6):
        try:execution.delete_pod(pod_id)
        except Exception as exc:errors.append(f'delete attempt {attempt}: {type(exc).__name__}')
        try:
            execution.get_pod(pod_id)
        except RunPodApiError as exc:
            cause=exc.__cause__
            missing=(isinstance(cause,urllib.error.HTTPError) and cause.code==404) or str(exc)=='unexpected get-Pod response status: 404'
            if missing:return dict(verified_absent=True,attempts=attempt,errors=errors)
            errors.append(f'verify attempt {attempt}: {type(exc).__name__}')
        except Exception as exc:
            errors.append(f'verify attempt {attempt}: {type(exc).__name__}')
        if attempt<5:time.sleep(1)
    return dict(verified_absent=False,attempts=5,errors=errors)


def execute(cfg,manifest,mode,*,execution,client,repo_sha,run_id):
    authorize(cfg,manifest)
    if mode=='evaluate' and cfg['governance'].get('paid_checkpoint_evaluation_allowed') is not True:
        raise PermissionError('V2 paid evaluation is not approved')
    if mode not in ('train','evaluate'):raise ValueError(mode)
    if not repo_sha or len(repo_sha)!=40 or not run_id or '/' in run_id or '..' in run_id:
        raise ValueError('safe run identity required')
    prefix=cfg['execution']['s3_prefix']+'/'+run_id
    suffix='training' if mode=='train' else 'checkpoints/step-096/evaluation'
    key=f'{prefix}/{suffix}/result.json'
    if infra.maybe_json(client,key) is not None:
        raise ValueError('run result already exists; refusing overwrite')
    if mode=='evaluate':
        training=infra.maybe_json(client,f'{prefix}/training/result.json')
        if not training or training.get('repo_sha')!=repo_sha or training.get('status')!='training_complete':
            raise ValueError('evaluation requires completed immutable training')
    body=request(cfg,repo_sha,run_id,mode,real_env=True)
    # Conditional creation prevents retries/concurrent launchers from buying the
    # same approved run twice. Unsupported conditional writes fail before allocation.
    claim=dict(run_id=run_id,repo_sha=repo_sha,mode=mode,manifest_sha256=__import__('hashlib').sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())
    client.put_object(Bucket=infra.required('RUNPOD_NETWORK_VOLUME_ID'),Key=cfg['execution']['s3_prefix']+'/approvals/'+cfg['governance']['budget_approval_id']+'/'+mode+'-claim.json',
                      Body=json.dumps(claim,sort_keys=True).encode(),IfNoneMatch='*')
    pod_id=None
    started=time.monotonic()
    record=dict(mode=mode,run_id=run_id,repo_sha=repo_sha,status='starting')
    wall=cfg['execution']['hard_wall_seconds_train' if mode=='train' else 'hard_wall_seconds_evaluate']
    try:
        pod=infra.create_with_capacity_retries(execution,body,1,0)
        pod_id=str(pod['id']);record['pod_id']=pod_id
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=wall:raise TimeoutError('V2 launcher hard wall reached')
            snapshot=execution.get_pod(pod_id)
            raw_rate=snapshot.get('adjustedCostPerHr',snapshot.get('costPerHr'))
            if raw_rate is not None:
                try:record['cost_per_hr']=float(raw_rate)
                except (ValueError,TypeError):pass
            record.update(bounded_cost(snapshot,elapsed,cfg,mode))
            bootstrap=infra.maybe_json(client,f'{prefix}/{suffix}/bootstrap/progress.json')
            if bootstrap and bootstrap.get('stage')=='bootstrap_failed':
                raise RuntimeError('V2 bootstrap failed; stop before consuming the run budget')
            terminal=infra.maybe_json(client,key)
            if terminal:
                if terminal.get('repo_sha')!=repo_sha or terminal.get('run_id')!=run_id:
                    raise ValueError('terminal evidence identity mismatch')
                if terminal.get('status')!=('training_complete' if mode=='train' else 'evaluation_complete'):
                    raise ValueError('remote V2 run failed')
                record.update(status=terminal['status'],result=terminal)
                return record
            if str(snapshot.get('desiredStatus','')).upper() in infra.TERMINAL:
                raise ValueError('pod stopped before complete evidence')
            time.sleep(cfg['execution']['poll_seconds'])
    finally:
        if pod_id:
            teardown=delete_verified(execution,pod_id)
            record['teardown']=teardown
            record['elapsed_including_teardown_seconds']=time.monotonic()-started
            if 'cost_per_hr' in record:
                record['estimated_cost_usd']=record['cost_per_hr']*record['elapsed_including_teardown_seconds']/3600
            print('EVALUATOR_SEMANTIC_V2_TEARDOWN '+json.dumps(dict(run_id=run_id,pod_id=pod_id,
                elapsed_including_teardown_seconds=record['elapsed_including_teardown_seconds'],
                estimated_cost_usd=record.get('estimated_cost_usd'),**teardown),sort_keys=True))
            if not teardown.get('verified_absent'):
                raise RuntimeError('V2 pod deletion could not be verified')


def s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client('s3',endpoint_url=infra.required('RUNPOD_S3_ENDPOINT_URL').rstrip('/'),
        region_name=infra.required('RUNPOD_DATACENTER_ID'),
        aws_access_key_id=infra.required('RUNPOD_S3_ACCESS_KEY_ID'),
        aws_secret_access_key=infra.required('RUNPOD_S3_SECRET_ACCESS_KEY'),
        config=Config(connect_timeout=5,read_timeout=5,retries={'mode':'standard','max_attempts':2}))


def quote_gpu(cfg):
    """Check the public catalog floor without allocating; the pod guard remains authoritative."""
    import urllib.request
    import urllib.error
    gpu=cfg['execution']['gpu_type_ids'][0]
    query='query { gpuTypes(input: { id: '+json.dumps(gpu)+' }) { id lowestPrice(input: { gpuCount: 1, secureCloud: true }) { stockStatus uninterruptablePrice } } }'
    request=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':query}).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+infra.required('RUNPOD_API_KEY'),
                 'User-Agent':'Mozilla/5.0 (compatible; Hephaestus-RunPod/1.0)'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=20) as response:payload=json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'catalog preflight HTTP {exc.code}') from None
    matches=payload.get('data',{}).get('gpuTypes',[])
    if payload.get('errors') or len(matches)!=1 or matches[0].get('id')!=gpu:
        raise ValueError('catalog preflight has no unambiguous GPU quote')
    price=matches[0].get('lowestPrice') or {}
    import math
    hourly=float(price.get('uninterruptablePrice') or 0)
    if not math.isfinite(hourly) or hourly<=0:
        raise ValueError('catalog preflight price is invalid')
    if hourly>cfg['execution']['max_hourly_usd_per_pod']:
        raise ValueError(f'catalog hourly price {hourly} exceeds approved {cfg["execution"]["max_hourly_usd_per_pod"]}; no pod allocated')
    if price.get('stockStatus')=='None':raise ValueError('catalog has no compatible GPU stock')
    return dict(gpu_type=gpu,catalog_floor_usd_per_hour=hourly,stock_status=price.get('stockStatus'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['train','evaluate'],required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    cfg,_,_,manifest=prepare('C')
    repo_sha=os.environ.get('GITHUB_SHA','0'*40)
    run_id=os.environ.get('HEPHAESTUS_EVALUATOR_V2_RUN_ID','evaluator-v2-C-dryrun')
    if not args.execute:
        print(json.dumps(dict(status='dry_run',paid_authorized=False,request=request(cfg,repo_sha,run_id,args.mode,real_env=False)),sort_keys=True,indent=2))
        return 0
    authorize(cfg,manifest)
    print('EVALUATOR_SEMANTIC_V2_PRICE_PREFLIGHT '+json.dumps(quote_gpu(cfg),sort_keys=True))
    record=execute(cfg,manifest,args.mode,
        execution=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider()),
        client=s3_client(),repo_sha=infra.required('GITHUB_SHA'),run_id=infra.required('HEPHAESTUS_EVALUATOR_V2_RUN_ID'))
    print('EVALUATOR_SEMANTIC_V2_LAUNCH '+json.dumps(record,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
