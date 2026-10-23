#!/usr/bin/env python3
"""Read-only regional RunPod GPU alternatives (never provisions a pod).

Read only GPU catalog stock and prices in the same DC as the approved volume.
No secret/provider response dumped into logs. Listed VRAM sizes come from
published GPU model specifications, not an API-provided measured device value.
"""
import json
import math
import os
import urllib.request

REGION="EU-CZ-1"
VRAM_GB={
  "NVIDIA RTX PRO 6000 Blackwell Server Edition":96,
  "NVIDIA RTX PRO 6000 Blackwell Workstation Edition":96,
  "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition":96,
  "NVIDIA A100 80GB PCIe":80,
  "NVIDIA A100-SXM4-80GB":80,
  "NVIDIA H100 PCIe":80,
  "NVIDIA H100 80GB HBM3":80,
  "NVIDIA L40S":48,
  "NVIDIA L40":48,
  "NVIDIA RTX 6000 Ada Generation":48,
  "NVIDIA RTX A6000":48,
  "NVIDIA A40":48,
}
def query():
    graphql=("query { gpuTypes { id lowestPrice(input: {gpuCount: 1, secureCloud: true, "
             'dataCenterId: "EU-CZ-1"}) { stockStatus uninterruptablePrice availableGpuCounts } } }')
    req=urllib.request.Request(
      "https://api.runpod.io/graphql",
      data=json.dumps({"query":graphql}).encode(),
      headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
               "Content-Type":"application/json",
               "User-Agent":"Hephaestus-ReadOnly-GPU-Discovery/1.0"},
      method="POST")
    with urllib.request.urlopen(req,timeout=25) as stream:
        return json.load(stream)

def select(payload):
    if not isinstance(payload,dict) or payload.get("errors"):
        raise ValueError("RunPod GPU catalog returned API errors")
    rows=(payload.get("data") or {}).get("gpuTypes")
    if not isinstance(rows,list):
        raise ValueError("RunPod GPU catalog missing types")
    found={}
    for r in rows:
        if not isinstance(r,dict) or r.get("id") not in VRAM_GB:
            continue
        name=r["id"]
        if name in found:
            raise ValueError("RunPod GPU catalog duplicate identity")
        quote=r.get("lowestPrice")
        status=quote.get("stockStatus") if isinstance(quote,dict) else None
        raw=quote.get("uninterruptablePrice") if isinstance(quote,dict) else None
        counts=quote.get("availableGpuCounts") if isinstance(quote,dict) else None
        price=float(raw) if type(raw) in (int,float) and math.isfinite(raw) and raw>0 else None
        counts_valid=counts is None or (
          isinstance(counts,list) and all(type(n)==int and n>0 for n in counts) and 1 in counts)
        possible=status in ("High","Medium","Low") and price is not None and counts_valid
        found[name]={"vram_gb":VRAM_GB[name],"status":status if status in ("High","Medium","Low","None") else "unknown",
              "usd_hourly":price,"one_gpu_confirmed":counts_valid,"eligible":possible}
    for name,vram in VRAM_GB.items():
        if name not in found:
            found[name]={"vram_gb":vram,"status":"absent","usd_hourly":None,
                        "one_gpu_confirmed":False,"eligible":False}
    return found

def main():
    try:
        found=select(query())
        print("D_GPU_ALTERNATES_EU_CZ_1 "+json.dumps({"region":REGION,"cloud":"Secure","gpu_count":1,
                "providers":found},sort_keys=True),flush=True)
    except Exception as ex:
        print("D_GPU_ALTERNATES_EU_CZ_1 "+json.dumps({"region":REGION,"error_type":type(ex).__name__,
                "safe_error":"no_usable_catalog_evidence"}),flush=True)
        raise SystemExit(75)

if __name__=="__main__":
    main()
