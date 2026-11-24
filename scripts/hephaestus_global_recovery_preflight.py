#!/usr/bin/env python3
"""Read-only RunPod + Czech S3 preflight. Never allocates compute or leaks credentials."""
from __future__ import annotations
import json, os
from decimal import Decimal
from urllib.request import Request, urlopen
import boto3

VOLUME='cmw51pjsk000007l9a0mv1jga'
SOURCE='cviwpryzao'
REQUIRED=['RUNPOD_API_KEY','RUNPOD_S3_ACCESS_KEY_ID','RUNPOD_S3_SECRET_ACCESS_KEY']


def request(url, *, body=None):
    raw=None if body is None else json.dumps(body).encode()
    headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'], 'Content-Type':'application/json'}
    with urlopen(Request(url,data=raw,headers=headers,method='GET' if raw is None else 'POST'),timeout=25) as resp:
        return json.load(resp)


def main():
    missing=[k for k in REQUIRED if not os.environ.get(k)]
    if missing: raise RuntimeError('Required GitHub Actions secrets absent: '+', '.join(missing))
    print('RUNPOD_SECRETS_PRESENT=true',flush=True)
    balance=request('https://api.runpod.io/graphql', body={'query':'query HephaestusReadOnlyBudget { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }'})
    user=balance.get('data',{}).get('myself',{})
    if balance.get('errors') or not user: raise RuntimeError('Authenticated RunPod account query failed')
    credit=Decimal(str(user['clientBalance']))
    if not credit.is_finite() or credit < 0: raise RuntimeError('Invalid RunPod credit balance')
    print('RUNPOD_BALANCE_POSITIVE='+str(credit>Decimal('1.0')).lower(),flush=True)
    print('RUNPOD_AUTOPAY_ENABLED='+str(user['isAutoPayEnabled']).lower(),flush=True)
    print('RUNPOD_SPEND_OBSERVED=true',flush=True)
    volumes=request('https://api.runpod.io/v2/global-volumes')
    if isinstance(volumes,dict): data=volumes.get('globalVolumes',volumes.get('global_volumes',volumes.get('data',volumes.get('volumes',[]))))
    elif isinstance(volumes,list): data=volumes
    else: raise RuntimeError('Unexpected global volume listing type')
    if not isinstance(data,list): raise RuntimeError('Unknown RunPod global volume listing shape; no paid action')
    matches=[v for v in data if isinstance(v,dict) and v.get('id')==VOLUME]
    if len(matches)!=1: raise RuntimeError('Exact Global Volume not identified; no paid action')
    print('EXPECTED_GLOBAL_VOLUME_EXISTS=true',flush=True)
    s3=boto3.client('s3',endpoint_url='https://s3api-eu-cz-1.runpod.io/', region_name='EU-CZ-1',aws_access_key_id=os.environ['RUNPOD_S3_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['RUNPOD_S3_SECRET_ACCESS_KEY'])
    s3.head_bucket(Bucket=SOURCE)
    listing=s3.list_objects_v2(Bucket=SOURCE,Prefix='hephaestus/scientific/',MaxKeys=3)
    if not listing.get('Contents'): raise RuntimeError('Original Czech scientific source has no listed objects')
    print('REGIONAL_S3_READ_ACCESS=true',flush=True)
    pods=request('https://api.runpod.io/v2/pods')
    if isinstance(pods,dict): entries=pods.get('pods',pods.get('data',[]))
    else: entries=pods
    if not isinstance(entries,list): raise RuntimeError('Unknown RunPod pods listing; allocation unsafe')
    conflicting=[x for x in entries if isinstance(x,dict) and str(x.get('name','')).startswith('hephaestus-hf-global-recovery-')]
    print('EXISTING_RECOVERY_POD_COUNT='+str(len(conflicting)),flush=True)
    if conflicting: raise RuntimeError('Existing recovery pod must be reconciled before creating another')
    print('PREFLIGHT_COMPLETE=true',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('PREFLIGHT_FAILED='+type(exc).__name__+': '+str(exc)[:300],flush=True)
        raise SystemExit(1)
