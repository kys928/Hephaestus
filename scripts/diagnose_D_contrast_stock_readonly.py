#!/usr/bin/env python3
"""Read-only RunPod regional GPU-catalog shape probe.

Logs ONLY whitelisted schema facts; no response dump, request headers,
credentials, tokens, account fields, or GPU provisioning.
"""
import json
import preflight_D_contrast_region_stock as gate

def shape(payload):
    out={"stage":"parsed","gpu":gate.GPU,"region":gate.DATACENTER}
    if not isinstance(payload,dict):
        return {**out,"shape_error":"root_not_object"}
    if payload.get("errors"):
        issues=payload["errors"]
        return {**out,"shape_error":"graphql_errors","errors_count":len(issues) if isinstance(issues,list) else None}
    d=payload.get("data")
    if not isinstance(d,dict):return {**out,"shape_error":"data_missing"}
    items=d.get("gpuTypes")
    if not isinstance(items,list):
        return {**out,"shape_error":"gpuTypes_not_list"}
    out["gpu_rows"]=len(items)
    if len(items)!=1:return {**out,"shape_error":"gpu_row_count_mismatch"}
    gpu=items[0]
    if not isinstance(gpu,dict):return {**out,"shape_error":"gpu_row_not_object"}
    out["id_matches_expected"]=gpu.get("id")==gate.GPU
    out["gpu_row_keys"]=sorted(k for k in gpu if k in ("id","displayName","lowestPrice","secureCloud"))
    price=gpu.get("lowestPrice")
    if price is None:return {**out,"shape_error":"lowestPrice_null","capacity_usable":False}
    if not isinstance(price,dict):return {**out,"shape_error":"lowestPrice_nonobject"}
    out["price_field_presence"]={k:k in price for k in ("stockStatus","uninterruptablePrice","availableGpuCounts")}
    status=price.get("stockStatus")
    out["stock_status"]=status if status in ("High","Medium","Low","None","HIGH","MEDIUM","LOW","NONE") else "unknown_or_null"
    value=price.get("uninterruptablePrice")
    out["price_type"]=type(value).__name__
    if type(value) in (int,float):out["price_at_most_2_49"]=(value<=gate.MAX_HOURLY_USD)
    counts=price.get("availableGpuCounts")
    out["gpu_counts_type"]=type(counts).__name__
    if isinstance(counts,list):
        out["gpu_counts_count"]=len(counts)
        out["gpu_counts_include_one"]=1 in counts
    return out

def main():
    try:
        payload=gate.query()
        detail=shape(payload)
        try:
            parsed=gate.inspect_response(payload)
            detail["existing_gate_result"]=parsed["confident_capacity_signal"]
        except (ValueError,TypeError,KeyError):
            detail["existing_gate_result"]="rejected"
    except Exception as exc:
        detail={"stage":"read_only_query_failed","exception_type":type(exc).__name__}
    print("D_CONTRAST_REGIONAL_STOCK_SAFE_DIAG "+json.dumps(detail,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
