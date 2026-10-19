#!/usr/bin/env python3
"""Zero-GPU, zero-claim RunPod stock gate for one preauthorized D contrast retry.

Only medium/high secure-cloud STOCK IN THE NETWORK VOLUME DATACENTER may
proceed; Low/None, quote drift, unknown schema or API error fail closed.
Exit 75 is a temporary no-capacity status suitable for future *job* reruns.
"""
import json
import math
import os
import sys
import urllib.request

GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
DATACENTER="EU-CZ-1"
MAX_HOURLY_USD=2.49

def inspect_response(payload):
    if not isinstance(payload,dict) or payload.get("errors"):
        raise ValueError("RunPod returned API errors or invalid response")
    items=(payload.get("data") or {}).get("gpuTypes")
    if not isinstance(items,list) or len(items)!=1 or items[0].get("id")!=GPU:
        raise ValueError("GPU identity ambiguous in RunPod catalog")
    quote=items[0].get("lowestPrice")
    if not isinstance(quote,dict):
        raise ValueError("data-center-specific secure-cloud quote missing")
    stock=quote.get("stockStatus")
    price=quote.get("uninterruptablePrice")
    counts=quote.get("availableGpuCounts")
    if stock not in ("High","Medium","Low","None"):
        raise ValueError("unrecognized stock status")
    if not isinstance(price,(int,float)) or not math.isfinite(price) or price<=0:
        raise ValueError("GPU hourly price unavailable")
    # RunPod's availableGpuCounts is optional in the published GraphQL
    # LowestPrice schema; live EU-CZ-1 responses may return JSON null.
    # The lowestPrice query itself is explicitly constrained to 1 GPU,
    # Secure Cloud, and the given datacenter. Its stockStatus is therefore
    # already the 1-GPU regional availability signal. Do not mistake null
    # for malformed provider data or a proof of zero GPUs.
    if counts is not None and (
        not isinstance(counts,list) or
        not all(type(n)==int and n>0 for n in counts)
    ):
        raise ValueError("invalid GPU-count metadata")
    one_gpu_confirmed=(counts is None or 1 in counts)
    suitable=(stock in ("High","Medium") and one_gpu_confirmed and price<=MAX_HOURLY_USD)
    return dict(gpu_type=GPU,datacenter=DATACENTER,secure_cloud=True,
        stock_status=stock,available_gpu_counts=counts,
        one_gpu_evidence=("regional_gpuCount_1_stock" if counts is None
                          else "regional_stock_and_explicit_gpu_counts"),
        quoted_usd_per_hour=float(price),max_hourly_usd=MAX_HOURLY_USD,
        confident_capacity_signal=suitable)

def query():
    key=os.environ["RUNPOD_API_KEY"]
    expression=("query { gpuTypes(input: { id: "+
        json.dumps(GPU)+
        " }) { id lowestPrice(input: {gpuCount: 1, secureCloud: true, dataCenterId: "+
        json.dumps(DATACENTER)+
        "}) { stockStatus uninterruptablePrice availableGpuCounts } } }")
    request=urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(dict(query=expression)).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,
                 "User-Agent":"Hephaestus-Capacity-Gate/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request,timeout=18) as stream:
        return json.load(stream)

# Error metadata is a closed vocabulary: never print a provider response,
# API token, request, exception body or arbitrary remote-supplied message.
SAFE_SCHEMA_CODES = {
    "RunPod returned API errors or invalid response": "graphql_error_response",
    "GPU identity ambiguous in RunPod catalog": "wrong_or_missing_gpu",
    "data-center-specific secure-cloud quote missing": "regional_quote_missing",
    "unrecognized stock status": "stock_status_unknown",
    "GPU hourly price unavailable": "price_unavailable",
    "invalid GPU-count metadata": "gpu_counts_invalid",
}

def main():
    try:
        status=inspect_response(query())
    except Exception as exc:
        safe={
            "confident_capacity_signal":False,
            "error_type":type(exc).__name__,
        }
        if isinstance(exc,ValueError):
            safe["error_code"]=SAFE_SCHEMA_CODES.get(str(exc),"provider_schema_invalid")
        else:
            safe["error_code"]="capacity_query_failed"
        print("D_CONTRAST_IN_REGION_CAPACITY "+json.dumps(safe,sort_keys=True),flush=True)
        return 75
    print("D_CONTRAST_IN_REGION_CAPACITY "+json.dumps(status,sort_keys=True),flush=True)
    return 0 if status["confident_capacity_signal"] else 75

if __name__=="__main__":
    raise SystemExit(main())
