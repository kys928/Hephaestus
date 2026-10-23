#!/usr/bin/env python3
"""Terminate exactly the user-specified stalled pod, verify deletion, no retries/launches."""
import os,json,time,urllib.request,urllib.error

POD="bnvazbgs2acrp4"
EXPECTED="hephaestus-D-counterfactuals-D-contrast-39974986184"
API="https://rest.runpod.io/v1/pods/"+POD
def req(method):
    request=urllib.request.Request(API,method=method,headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
        "Accept":"application/json","User-Agent":"Hephaestus-UserRequested-Pod-Termination"})
    try:
        with urllib.request.urlopen(request,timeout=30) as r:return r.status,json.load(r) if method=="GET" else None
    except urllib.error.HTTPError as e:
        if e.code==404:return 404,None
        return e.code,None

def main():
    code,pod=req("GET")
    if code==404:
        print("D_A40_STOP "+json.dumps({"pod_id":POD,"verified_absent":True,"already_absent":True}),flush=True)
        return
    if code!=200 or not isinstance(pod,dict):
        raise RuntimeError("refusing deletion: could not verify pod identity")
    if pod.get("id")!=POD or pod.get("name")!=EXPECTED:
        raise RuntimeError("refusing deletion: expected experiment pod mismatch")
    print("D_A40_STOP "+json.dumps({"pod_id":POD,"verified_identity":True,"before":pod.get("desiredStatus")}),flush=True)
    status,_=req("DELETE")
    if status not in (200,202,204,404):
        raise RuntimeError("RunPod delete request failed with HTTP status "+str(status))
    for attempt in range(1,19):
        status,_=req("GET")
        if status==404:
            print("D_A40_STOP "+json.dumps({"pod_id":POD,"verified_absent":True,
                "delete_http_status":status,"verify_attempt":attempt}),flush=True)
            return
        time.sleep(3)
    raise RuntimeError("pod deletion NOT confirmed; manual stop/verification needed")
if __name__=="__main__":main()
