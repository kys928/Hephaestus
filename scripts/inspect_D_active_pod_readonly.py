#!/usr/bin/env python3
"""Read-only metadata for the exact user-observed RunPod A40 pod."""
import json,os,urllib.request
POD="bnvazbgs2acrp4"
def main():
    r=urllib.request.Request("https://rest.runpod.io/v1/pods/"+POD,
        headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
                 "Accept":"application/json","User-Agent":"Hephaestus-ReadOnly-Live-D"})
    try:
        with urllib.request.urlopen(r,timeout=20) as o: data=json.load(o)
        if not isinstance(data,dict):raise ValueError("unexpected shape")
        cleaned={k:data.get(k) for k in ("id","name","desiredStatus","costPerHr","adjustedCostPerHr",
             "gpuCount","gpuTypeId","cloudType","dataCenterId","createdAt","lastStatusChange","uptimeSeconds")
             if k in data}
        cleaned["has_runtime_metadata"]=isinstance(data.get("runtime"),dict)
        print("D_ACTIVE_POD_READONLY "+json.dumps(cleaned,sort_keys=True),flush=True)
    except Exception as e:
        print("D_ACTIVE_POD_READONLY "+json.dumps({"status":"read_failed","error_type":type(e).__name__}),flush=True)
        raise SystemExit(75)
if __name__=="__main__":main()
