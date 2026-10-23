#!/usr/bin/env python3
"""Live READ ONLY RunPod S3 evidence inspection for the ACTIVE D contrast pod.

All remote S3 operations: ListObjectsV2, HeadObject, GetObject.
NEVER creates, writes, modifies S3, touches a pod, or prints credentials.
"""
import json,datetime
import run_interface_repair_v1 as s3
from evaluator_contrast_inference_protocol import PREFIX

RUN_ID="D-contrast-39974986184"
ROOT=f"{PREFIX}/{RUN_ID}/evaluation"
APPROVAL=f"{PREFIX}/approvals/heph-D-contrasts-A40-EUSE1-20261023-ONE-PAID-ATTEMPT/once.json"
KEYS=[ROOT+"/bootstrap/progress.json",ROOT+"/progress.json",
      ROOT+"/manifest.json",ROOT+"/result.json",
      ROOT+"/timing/latest.json",APPROVAL]
MAX_BYTES=1000000
def utc(t):
    if t is None: return None
    return t.isoformat() if hasattr(t,"isoformat") else str(t)
def missing(e):
    v=getattr(e,"response",{})
    return isinstance(v,dict) and str(v.get("Error",{}).get("Code")) in ("404","NoSuchKey","NotFound")
def get_json(c,bucket,k):
    try:
        head=c.head_object(Bucket=bucket,Key=k)
        length=int(head.get("ContentLength",-1))
        if length<0 or length>MAX_BYTES:return {"present":True,"read":"size_rejected","bytes":length}
        obj=c.get_object(Bucket=bucket,Key=k)
        try: raw=obj["Body"].read(MAX_BYTES+1)
        finally: obj["Body"].close()
        if len(raw)!=length:return {"present":True,"read":"size_drift"}
        j=json.loads(raw)
        return {"present":True,"last_modified":utc(head.get("LastModified")),
                "data":j}
    except Exception as e:
        if missing(e):return {"present":False}
        return {"present":None,"error_type":type(e).__name__}
def listing(c,bucket,prefix):
    token=None;rows=[]
    for _ in range(20):
        kw=dict(Bucket=bucket,Prefix=prefix,MaxKeys=250)
        if token:kw["ContinuationToken"]=token
        obj=c.list_objects_v2(**kw)
        for i in obj.get("Contents",[]):
            rows.append({"name":i["Key"].split("/")[-1],
                         "key":i["Key"],"bytes":i["Size"],
                         "modified":utc(i.get("LastModified"))})
        token=obj.get("NextContinuationToken")
        if not token:break
    return rows
def sanitized(records):
    out={}
    for key,record in records.items():
        value=record.get("data")
        meta={k:v for k,v in record.items() if k!="data"}
        if isinstance(value,dict):
            if key=="bootstrap":
                meta["state"]={k:value.get(k) for k in ("stage","timestamp_unix","venv_ready","hf_cache_root_exists","nvidia_smi_error")}
            elif key=="progress":
                meta["state"]={k:value.get(k) for k in ("stage","completed","total","total_generation_seconds","valid_exploratory")}
            elif key=="timing":
                meta["state"]={k:value.get(k) for k in ("stage","sequence","utc","elapsed_monotonic_seconds","details")}
            elif key=="result":
                meta["state"]={k:value.get(k) for k in ("status","run_id","error_type","unique_prompts","paired_challenges","training_performed")}
            elif key=="manifest":
                meta["state"]={k:value.get(k) for k in ("protocol","run_id","D_checkpoint_sha256","unique_prompts","paired_challenges")}
            elif key=="claim":
                meta["state"]={k:value.get(k) for k in ("runid","protocol","gpu_type_id")}
        out[key]=meta
    return out
def main():
    c=s3.s3_client();bucket=s3.bucket()
    records={}
    for label,key in zip(("bootstrap","progress","manifest","result","timing","claim"),KEYS):
        records[label]=get_json(c,bucket,key)
    all_files=listing(c,bucket,ROOT+"/")
    events=sorted((x for x in all_files if "/timing/events/" in x["key"]),key=lambda x:x["key"])
    samples=[x for x in all_files if "/samples/" in x["key"]]
    stage_receipts=[]
    for x in events[-45:]:
        v=get_json(c,bucket,x["key"])
        d=v.get("data")
        if isinstance(d,dict):
            stage_receipts.append({k:d.get(k) for k in ("sequence","stage","utc","elapsed_monotonic_seconds","details")})
    possible_error=None
    if records["claim"]["present"] and not records["timing"]["present"] and not records["bootstrap"]["present"]:
        possible_error="Pod may still be in shell/bootstrap; without cross-region volume mount its bootstrap file is LOCAL and not remotely visible. The S3 trace begins only after Python D runner starts."
    report={"run_id":RUN_ID,
            "observed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "read_only":True,
            "files_listed":len(all_files),
            "recent_objects":sorted(all_files,key=lambda x:x["modified"],reverse=True)[:25],
            "records":sanitized(records),
            "timing_event_count":len(events),
            "timing_stage_history":stage_receipts,
            "sample_shard_count":len(samples),
            "sample_shard_files":[{"name":x["name"],"bytes":x["bytes"],"modified":x["modified"]} for x in samples],
            "note":possible_error}
    print("D_ACTIVE_S3_READONLY_STATUS "+json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":main()
