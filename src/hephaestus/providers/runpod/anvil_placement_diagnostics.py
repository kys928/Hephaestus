"""Read-only placement diagnostics; never expose provider bodies or credentials."""
from datetime import datetime, timezone
from time import sleep
from urllib.error import URLError
from hephaestus.providers.runpod.execution import RunPodApiError, AmbiguousRunPodOperation
from hephaestus.providers.runpod.anvil_live_lease import GPU, GLOBAL, MAX_RATE, DISK_ALLOWANCE, LIVE_LEASE_SECONDS, fundable

STOCK = {"LOW", "MEDIUM", "HIGH"}


class PlacementFailure(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__("placement gate failed: " + diagnostic["reason"])
        self.diagnostic = diagnostic


def safe_failure(stage, error):
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    code = code if type(code) is int and 100 <= code <= 599 else None
    if isinstance(error, AmbiguousRunPodOperation):
        reason = "allocation_outcome_unknown_reconcile_no_retry"
    elif code in (401,403):
        reason = "authentication_or_permission"
    elif code == 429:
        reason = "rate_limit"
    elif code is not None and code >= 500:
        reason = "provider_server_error"
    elif isinstance(error, PermissionError):
        reason = "permission_or_policy_gate"
    elif isinstance(error, (TimeoutError, URLError, OSError)):
        reason = "transport_error"
    elif isinstance(error, RunPodApiError) and str(error) == "No allowed GPU available in the pinned datacenter; no allocation attempted":
        reason = "no_allowed_stock"
    elif isinstance(error, RunPodApiError) and str(error) == "GPU placement catalog unavailable; no allocation attempted":
        reason = "catalog_identity_or_schema"
    else:
        reason = "operation_failed"
    return {"stage":stage,"reason":reason,"exception":type(error).__name__,"http_status":code}


def inspect_placement(pricing, provider, *, region, report, gpu=GPU):
    """Observe and retain exact quote, stock, mount and budget gates separately."""
    common={"schema":"anvil.placement-diagnostic.v1","region":region,"gpu":gpu,
            "cloud":"SECURE","global_volume":GLOBAL,"allocation_attempted":False}
    def emit(stage, status, **extra):
        doc=dict(common,stage=stage,status=status,observed_at=datetime.now(timezone.utc).isoformat(),**extra)
        report(doc)
        return doc
    def fail(stage, reason, **extra):
        doc=emit(stage,"blocked",reason=reason,**extra)
        raise PlacementFailure(doc)
    def observe(stage, call):
        try:
            return call()
        except Exception as error:
            detail=safe_failure(stage,error)
            detail.pop("stage")
            fail(stage,**detail)
    q=observe("gpu_quote",lambda:pricing.quote_gpu(gpu_id=gpu,datacenter_id=region))
    total=q.price_usd_per_hour+DISK_ALLOWANCE
    emit("gpu_quote","observed",gpu_hour_usd=str(q.price_usd_per_hour),
         vram_gb=q.gpu_memory_gb,all_in_ceiling_usd=str(total))
    if q.gpu_memory_gb<48:
        fail("gpu_quote","insufficient_vram")
    if total>MAX_RATE:
        fail("gpu_quote","price_ceiling_exceeded",all_in_ceiling_usd=str(total))
    status,catalog=observe("regional_catalog",lambda:provider._request("GET","catalog/datacenters/"+region+"?include=GPU_AVAILABILITY"))
    if status!=200 or not isinstance(catalog,dict) or catalog.get("id")!=region or not isinstance(catalog.get("gpuAvailability"),list):
        fail("regional_catalog","catalog_identity_or_schema",http_status=status if type(status) is int else None,
             response_type=type(catalog).__name__,
             identity_matches=isinstance(catalog,dict) and catalog.get("id")==region,
             availability_present=isinstance(catalog,dict) and "gpuAvailability" in catalog,
             availability_type=type(catalog.get("gpuAvailability")).__name__ if isinstance(catalog,dict) else "unavailable")
    rows=[r for r in catalog["gpuAvailability"] if isinstance(r,dict) and r.get("id")==gpu]
    stock=[r.get("availability") if r.get("availability") in STOCK|{"NONE","UNAVAILABLE"} else "UNKNOWN" for r in rows]
    emit("regional_catalog","observed",exact_sku_rows=len(rows),stock=stock)
    if len(rows)>1:
        fail("regional_catalog","ambiguous_sku_rows")
    if not rows or stock[0] in {"NONE","UNAVAILABLE"}:
        fail("regional_catalog","no_allowed_stock",exact_sku_rows=len(rows),stock=stock)
    if stock[0] not in STOCK:
        fail("regional_catalog","unknown_stock_status",stock=stock)
    status,volume=observe("global_volume",lambda:provider._request("GET","global-volumes/"+GLOBAL))
    if status!=200 or not isinstance(volume,dict) or volume.get("id")!=GLOBAL:
        fail("global_volume","volume_identity_or_schema",http_status=status if type(status) is int else None)
    emit("global_volume","verified",identity_confirmed=True)
    account=observe("prepaid_budget",pricing.account)
    if account.autopay_enabled:
        fail("prepaid_budget","autopay_enabled")
    if not fundable(account,total):
        fail("prepaid_budget","insufficient_prepaid_funding")
    emit("prepaid_budget","verified",fundable=True)
    selected=observe("final_stock_recheck",lambda:provider.select_gpu([gpu]))
    if selected!=gpu:
        fail("final_stock_recheck","selected_gpu_mismatch")
    emit("final_stock_recheck","verified",exact_gpu_selected=True)
    return {"schema":"anvil.first-live-price.v1","gpu":gpu,"region":region,
            "gpu_hour_usd":str(q.price_usd_per_hour),"disk_hour_allowance_usd":str(DISK_ALLOWANCE),
            "all_in_hour_ceiling_usd":str(total),"maximum_wall_seconds":LIVE_LEASE_SECONDS,
            "existing_volume":GLOBAL,"disk_price_source":"https://docs.runpod.io/pods/pricing",
            "observed_at":datetime.now(timezone.utc).isoformat(),"allocation_attempted":False}

# Rechecks are strictly read-only. A missing availability field never counts as
# stock, and an explicit wrong datacenter identity never authorizes a retry.
def retryable_placement(error):
    d=error.diagnostic
    if d["stage"] not in {"regional_catalog","final_stock_recheck"}:
        return False
    if d["reason"]=="catalog_identity_or_schema":
        return d.get("identity_matches") is True or d["stage"]=="final_stock_recheck"
    return d["reason"] in {"no_allowed_stock","unknown_stock_status","rate_limit",
                            "provider_server_error","transport_error"}


def inspect_with_rechecks(pricing,provider,*,region,gpu,report,pause=sleep,attempts=3):
    if type(attempts) is not int or not 1<=attempts<=3:
        raise ValueError("placement recheck count outside fixed bound")
    for n in range(1,attempts+1):
        try:
            return inspect_placement(pricing,provider,region=region,gpu=gpu,
                                     report=lambda d:report(dict(d,recheck_attempt=n)))
        except PlacementFailure as e:
            if n==attempts or not retryable_placement(e):
                raise
            pause(2*n)


def recheck_exact_stock(provider,gpu,*,report,pause=sleep,attempts=3):
    """Immediately before durable intent; no create or mutation on this path."""
    if type(attempts) is not int or not 1<=attempts<=3:
        raise ValueError("final stock recheck count outside fixed bound")
    for n in range(1,attempts+1):
        try:
            selected=provider.select_gpu([gpu])
            if selected!=gpu:
                raise PermissionError("selected GPU identity differs")
            report({"schema":"anvil.pre-intent-stock.v1","attempt":n,"gpu":gpu,
                    "status":"verified","allocation_attempted":False})
            return selected
        except Exception as e:
            d=safe_failure("final_stock_recheck",e)
            report(dict(d,schema="anvil.pre-intent-stock.v1",attempt=n,gpu=gpu,
                        status="blocked",allocation_attempted=False))
            if n==attempts or not retryable_placement(PlacementFailure(d)):
                raise
            pause(2*n)
