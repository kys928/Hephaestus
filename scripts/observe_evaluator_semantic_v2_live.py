"""Read-only snapshot of the approved C run; never allocates or writes S3."""
import json,os
import run_interface_repair_v1 as v1
import launch_interface_repair_v2 as infra
from launch_evaluator_semantic_repair_v2 import s3_client
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
run_id='evaluator-semantic-v2-C-38935682810'
root='hephaestus/scientific/v5/evaluator_semantic_repair_v2/'+run_id
client=s3_client()
execution=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider())
_,payload=execution._request('GET','pods')
pods=payload if isinstance(payload,list) else payload.get('pods',payload.get('data',[]))
fields=('id','name','desiredStatus','costPerHr','adjustedCostPerHr','gpuTypeId','gpuDisplayName','createdAt')
safe=[{k:p.get(k) for k in fields if k in p} for p in pods if run_id in str(p.get('name',''))]
records={}
for suffix in ('training/bootstrap/progress.json','training/progress.json','training/result.json','checkpoints/step-096/evaluation/bootstrap/progress.json','checkpoints/step-096/evaluation/progress.json','checkpoints/step-096/evaluation/result.json'):
    records[suffix]=infra.maybe_json(client,root+'/'+suffix)
print('V2_LIVE_OBSERVATION '+json.dumps(dict(run_id=run_id,pods=safe,records=records,pod_allocations=0,s3_writes=0),sort_keys=True))

if (records.get('checkpoints/step-096/evaluation/result.json') or {}).get('status')=='evaluation_complete':
    from assess_evaluator_semantic_v2_run import assess
    assess()
