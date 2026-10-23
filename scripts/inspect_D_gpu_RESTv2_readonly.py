#!/usr/bin/env python3
"""Read-only REST v2 RunPod GPU and per-DC availability inventory."""
import os,json,urllib.request
URL="https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD"
def main():
    req=urllib.request.Request(URL,headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
        "Accept":"application/json","User-Agent":"Hephaestus-ReadOnly-Catalog-V2"})
    try:
        with urllib.request.urlopen(req,timeout=25) as f: data=json.load(f)
        root=data if isinstance(data,dict) else {}
        rows=(root.get("gpus") or root.get("data") or root.get("gpuTypes") or [])
        if isinstance(rows,dict):
            rows=rows.get("gpus",[]) or rows.get("items",[])
        wanted=("A40","A6000","L40","RTX 6000 Ada","A100","H100","RTX PRO 6000","H200")
        found=[]
        for r in rows if isinstance(rows,list) else []:
            if not isinstance(r,dict):continue
            name=str(r.get("id",r.get("name","")))
            if not any(w in name for w in wanted):continue
            dc=r.get("dataCenters") or r.get("datacenters") or []
            centers=[]
            for x in dc if isinstance(dc,list) else []:
                if isinstance(x,dict):
                    centers.append({k:x.get(k) for k in ("id","availability","stockStatus") if k in x})
            found.append({"id":name,"memory":r.get("memory"),
                 "price":r.get("price"),"availability":r.get("availability"),
                 "data_centers":centers[:30]})
        print("D_REST_V2_GPU_CATALOG "+json.dumps({"response_type":type(data).__name__,
            "root_keys":sorted(root.keys()),"row_count":len(rows) if isinstance(rows,list) else None,
            "selected":found},sort_keys=True),flush=True)
    except Exception as ex:
        print("D_REST_V2_GPU_CATALOG "+json.dumps({"error_type":type(ex).__name__,
        "state":"catalog_query_failed"}),flush=True)
        raise SystemExit(75)
if __name__=="__main__":main()
