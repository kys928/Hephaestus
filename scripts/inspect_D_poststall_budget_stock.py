#!/usr/bin/env python3
"""Read-only post-stall RunPod balance bracket + 48GB+ Secure Cloud GPU stock.
No API secret, balance number or account info exposed.
"""
import json,os,urllib.request,math

APIKEY=os.environ["RUNPOD_API_KEY"]
def get_json(url):
 req=urllib.request.Request(url,headers={"Authorization":"Bearer "+APIKEY,
   "Accept":"application/json","User-Agent":"Hephaestus-PostStall-ReadOnly"})
 with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
def gql(q):
 req=urllib.request.Request("https://api.runpod.io/graphql",method="POST",
     headers={"Authorization":"Bearer "+APIKEY,"Content-Type":"application/json"},
     data=json.dumps({"query":q}).encode())
 with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
def main():
  try:
    account=gql("query { myself { clientBalance } }")
    myself=(account.get("data") or {}).get("myself")
    value=myself.get("clientBalance") if isinstance(myself,dict) else None
    if type(value) not in (int,float) or not math.isfinite(value):
      raise ValueError("balance missing")
    print("D_RETRY_BUDGET "+json.dumps({"balance_read_ok":True,
       "at_least_usd":{str(x):value>=x for x in (0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0)}}),flush=True)
  except Exception as exc:
    print("D_RETRY_BUDGET "+json.dumps({"balance_read_ok":False,"error_type":type(exc).__name__}),flush=True)
  try:
    catalog=get_json("https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD")
    rows=catalog.get("gpus") or []
    allowed=("NVIDIA RTX PRO 6000 Blackwell Server Edition",
             "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
             "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
             "NVIDIA RTX 6000 Ada Generation","NVIDIA RTX A6000","NVIDIA A40",
             "NVIDIA L40","NVIDIA L40S","NVIDIA A100-SXM4-80GB",
             "NVIDIA A100 80GB PCIe")
    alternatives=[]
    for r in rows:
      if r.get("id") not in allowed:continue
      for dc in r.get("dataCenters") or []:
       if dc.get("availability") not in ("LOW","MEDIUM","HIGH"):continue
       alternatives.append(dict(gpu=r["id"],vram_gb=r.get("memory"),
          dc=dc.get("id"),stock=dc.get("availability"),catalog_secure_price=(r.get("price") or {}).get("secure")))
    alternatives.sort(key=lambda x:(0 if "RTX PRO 6000 Blackwell Server Edition"==x["gpu"] else 1,
       (x["catalog_secure_price"] if type(x["catalog_secure_price"]) in (float,int) else 999)))
    print("D_RETRY_REST_GPU_CATALOG "+json.dumps({"candidates":alternatives[:35],"count":len(alternatives)}),flush=True)
    # Confirm exact one-GPU Secure Cloud stock. GraphQL stock may differ from
    # catalog availability: only a clean positive scoped quote is eligible.
    best=alternatives[:25]
    fields=["q"+str(i)+":gpuTypes(input:{id:"+json.dumps(r["gpu"])+
       "}){id lowestPrice(input:{gpuCount:1,secureCloud:true,dataCenterId:"+
       json.dumps(r["dc"])+
       "}){stockStatus uninterruptablePrice availableGpuCounts}}" for i,r in enumerate(best)]
    if fields:
      result=gql("query { "+" ".join(fields)+" }")
      if result.get("errors"):raise ValueError("scoped GPU quote errors")
      data=result.get("data") or {}
      eligible=[]
      for i,item in enumerate(best):
       rows=data.get("q"+str(i))
       if not isinstance(rows,list) or len(rows)!=1 or rows[0].get("id")!=item["gpu"]:continue
       q=rows[0].get("lowestPrice") or {}
       stock=q.get("stockStatus");price=q.get("uninterruptablePrice")
       counts=q.get("availableGpuCounts")
       if stock not in ("Low","Medium","High") or type(price) not in (int,float) or not math.isfinite(price) or price<=0:continue
       if counts is not None and (not isinstance(counts,list) or 1 not in counts):continue
       eligible.append({**item,"verified_secure_price":price,"verified_stock":stock})
      print("D_RETRY_VERIFIED_OFFERS "+json.dumps({"offers":eligible}),flush=True)
    else:print("D_RETRY_VERIFIED_OFFERS "+json.dumps({"offers":[]}),flush=True)
  except Exception as exc:
    print("D_RETRY_VERIFIED_OFFERS "+json.dumps({"offers":[],"error_type":type(exc).__name__}),flush=True)
if __name__=="__main__":main()
