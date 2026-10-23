#!/usr/bin/env python3
"""Specific per-GPU, per-DC secure 1x stock quotes; absolutely read-only."""
import os,json,urllib.request,math
CANDIDATES=[
("NVIDIA A40","EU-SE-1",48),
("NVIDIA RTX A6000","CA-MTL-3",48),
("NVIDIA L40","US-KS-2",48),
("NVIDIA L40S","US-NC-1",48),
("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb","EUR-IS-2",48),
("NVIDIA RTX PRO 6000 Blackwell Workstation Edition","EU-RO-1",96),
("NVIDIA RTX PRO 6000 Blackwell Server Edition","EUR-IS-2",96),
]
def main():
 fields=[]
 for i,(gpu,dc,_) in enumerate(CANDIDATES):
  fields.append(f'q{i}:gpuTypes(input:{{id:{json.dumps(gpu)}}})'+
    f'{{id lowestPrice(input:{{gpuCount:1,secureCloud:true,dataCenterId:{json.dumps(dc)}}})'+
    '{stockStatus uninterruptablePrice availableGpuCounts}}')
 query='query {'+" ".join(fields)+'}'
 request=urllib.request.Request("https://api.runpod.io/graphql",
     data=json.dumps({"query":query}).encode(),
     headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
              "Content-Type":"application/json","User-Agent":"Hephaestus-ExactSecure-ReadOnly/1.0"},
     method="POST")
 try:
  with urllib.request.urlopen(request,timeout=30) as r: payload=json.load(r)
  if payload.get("errors"): raise ValueError("GraphQL errors")
  data=payload.get("data")
  if not isinstance(data,dict):raise ValueError("no data")
  offers=[]
  for i,(gpu,dc,vr) in enumerate(CANDIDATES):
   rows=data.get('q'+str(i))
   if not isinstance(rows,list) or len(rows)!=1 or rows[0].get("id")!=gpu:continue
   quote=rows[0].get("lowestPrice")
   if not isinstance(quote,dict):continue
   status=quote.get("stockStatus")
   price=quote.get("uninterruptablePrice")
   counts=quote.get("availableGpuCounts")
   valid=status in ("Low","Medium","High") and type(price) in (float,int) and math.isfinite(price) and price>0
   if counts is not None:valid=valid and isinstance(counts,list) and 1 in counts
   offers.append({"gpu":gpu,"gpu_region":dc,"vram_gb":vr,
           "status":status if status in ("Low","Medium","High","None") else "unknown",
           "secure_usd_hour":price if type(price) in (float,int) else None,
           "eligible":bool(valid),"gpu_count":1})
  print("D_EXACT_SECURE_QUOTES "+json.dumps({"offers":offers},sort_keys=True),flush=True)
 except Exception as e:
  print("D_EXACT_SECURE_QUOTES "+json.dumps({"error_type":type(e).__name__,"offers":[]}),flush=True)
  raise SystemExit(75)
if __name__=="__main__":main()
