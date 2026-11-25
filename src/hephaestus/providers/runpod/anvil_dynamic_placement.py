"""Rank Secure Cloud Global Volume placements without allocating resources."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hephaestus.providers.runpod.anvil_hardware import GPU_NAMES, valid_region
from hephaestus.providers.runpod.anvil_live_lease import GLOBAL, MAX_RATE, DISK_ALLOWANCE
from hephaestus.providers.runpod.cost_intelligence import REST_URL, usd
from hephaestus.providers.runpod.anvil_placement_diagnostics import safe_failure

API="https://api.runpod.io/v2"
STOCK={"HIGH":0,"MEDIUM":1,"LOW":2}

def discover(pricing,report=lambda d:None):
    volume=pricing._request(API+"/global-volumes/"+GLOBAL,"GET",None)
    if not isinstance(volume,dict) or volume.get("id")!=GLOBAL:
        raise PermissionError("Global Volume identity differs")
    account=pricing.account()
    if account.autopay_enabled:
        raise PermissionError("autopay must remain disabled")
    catalog=pricing._request(REST_URL,"GET",None)["gpus"]
    eligible={}
    for row in catalog:
        sku=row.get("id")
        if sku not in GPU_NAMES or row.get("secure") is not True or type(row.get("memory")) is not int or row["memory"]<48:
            continue
        try: rate=usd(row["price"]["secure"])
        except (KeyError,TypeError,ValueError): continue
        if 0<rate and rate+DISK_ALLOWANCE<=MAX_RATE and account.credit_balance_usd>rate+DISK_ALLOWANCE+account.already_spending_usd_per_hour:
            eligible[sku]=(rate,row["memory"])
    raw=pricing._request(API+"/catalog/datacenters","GET",None)
    rows=raw.get("dataCenters",raw.get("datacenters",[])) if isinstance(raw,dict) else raw
    regions=sorted({r["id"] for r in rows if isinstance(r,dict) and valid_region(r.get("id"))})
    def observe(region):
        try:
            data=pricing._request(API+"/catalog/datacenters/"+region+"?include=GPU_AVAILABILITY","GET",None)
            if data.get("id")!=region or not isinstance(data.get("gpuAvailability"),list):
                raise ValueError("invalid regional catalog")
            result=[]
            for sku,(rate,memory) in eligible.items():
                matches=[r for r in data["gpuAvailability"] if isinstance(r,dict) and r.get("id")==sku]
                if len(matches)==1 and matches[0].get("availability") in STOCK:
                    result.append({"region":region,"gpu":sku,"stock":matches[0]["availability"],"vram_gb":memory,
                        "gpu_hour_usd":str(rate),"all_in_hour_ceiling_usd":str(rate+DISK_ALLOWANCE)})
            return result,None
        except Exception as e:
            return [],dict(safe_failure("dynamic_region_catalog",e),region=region)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(observe,regions))
    candidates=[r for group,_ in results for r in group]
    candidates.sort(key=lambda r:(STOCK[r["stock"]],Decimal(r["all_in_hour_ceiling_usd"]),r["region"]=="EU-SE-1",r["region"],r["gpu"]))
    report({"schema":"anvil.dynamic-placement.v1","eligible_skus":sorted(eligible),"regions_checked":len(regions),
        "candidates":candidates,"unverified":[e for _,e in results if e],"allocation_attempted":False})
    return candidates
