#!/usr/bin/env python3
"""Read-only multi-region RunPod stock; S3 evidence remains in EU-CZ-1.

No pod allocation, S3 writes, billing action or new network volume.
"""
import json,os,urllib.request,math

REGIONS=["EU-RO-1","EU-NL-1","EU-FR-1","EUR-IS-1","EU-CZ-1",
         "US-KS-2","US-GA-1","US-TX-1"]
VRAM={
"NVIDIA A40":48,"NVIDIA RTX A6000":48,
"NVIDIA RTX 6000 Ada Generation":48,"NVIDIA L40S":48,
"NVIDIA L40":48,"NVIDIA A100 80GB PCIe":80,
"NVIDIA A100-SXM4-80GB":80,"NVIDIA RTX PRO 6000 Blackwell Server Edition":96,
"NVIDIA RTX PRO 6000 Blackwell Workstation Edition":96,
}
ALIASES={f"q{i}":region for i,region in enumerate(REGIONS)}
def query():
    fields=" ".join(f'{alias}: lowestPrice(input: {{gpuCount: 1, secureCloud: true, dataCenterId: "{region}"}}) {{ stockStatus uninterruptablePrice availableGpuCounts }}'
                   for alias,region in ALIASES.items())
    expression="query { gpuTypes { id "+fields+" } }"
    req=urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query":expression}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
                 "User-Agent":"Hephaestus-MultiRegion-ReadOnly/1.0"},method="POST")
    with urllib.request.urlopen(req,timeout=30) as response:
        return json.load(response)

def interpret(payload):
    if not isinstance(payload,dict) or payload.get("errors"):raise ValueError("api response not clean")
    rows=(payload.get("data") or {}).get("gpuTypes")
    if not isinstance(rows,list):raise ValueError("gpu types invalid")
    prices=[]
    seen=set()
    for row in rows:
        if not isinstance(row,dict):continue
        gpu=row.get("id")
        if gpu not in VRAM:continue
        if gpu in seen:raise ValueError("duplicate GPU")
        seen.add(gpu)
        for alias,region in ALIASES.items():
            q=row.get(alias)
            if not isinstance(q,dict):continue
            s=q.get("stockStatus")
            p=q.get("uninterruptablePrice")
            n=q.get("availableGpuCounts")
            if s not in ("High","Medium","Low") or type(p) not in (float,int) or not math.isfinite(p) or p<=0:continue
            if n is not None and (not isinstance(n,list) or 1 not in n):continue
            prices.append(dict(gpu=gpu,vram_gb=VRAM[gpu],gpu_datacenter=region,
                   cloud="SECURE",gpu_count=1,stock=s,hourly_usd=p,estimated_max_45m_usd=round(p*0.75,3)))
    return sorted(prices,key=lambda x:(0 if x["gpu_datacenter"].startswith(("EU-","EUR-")) else 1,
       {"High":0,"Medium":1,"Low":2}[x["stock"]],x["hourly_usd"],-x["vram_gb"]))

if __name__=="__main__":
    try:
        offers=interpret(query())
        print("D_GPU_MULTIREGION_STOCK "+json.dumps({"region_count":len(REGIONS),"offers":offers},sort_keys=True))
    except Exception as exc:
        print("D_GPU_MULTIREGION_STOCK "+json.dumps({"error_type":type(exc).__name__,"offers":[]}))
        raise SystemExit(75)
