"""Read-only placement diagnostics; never expose provider bodies or credentials."""
from datetime import datetime, timezone
from urllib.error import URLError
from hephaestus.providers.runpod.execution import RunPodApiError, AmbiguousRunPodOperation
from hephaestus.providers.runpod.anvil_live_lease import GPU, GLOBAL, MAX_RATE, DISK_ALLOWANCE

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
    if account.credit_balance_usd<=total+account.already_spending_usd_per_hour:
        fail("prepaid_budget","insufficient_prepaid_funding")
    emit("prepaid_budget","verified",fundable=True)
    selected=observe("final_stock_recheck",lambda:provider.select_gpu([gpu]))
    if selected!=gpu:
        fail("final_stock_recheck","selected_gpu_mismatch")
    emit("final_stock_recheck","verified",exact_gpu_selected=True)
    return {"schema":"anvil.first-live-price.v1","gpu":gpu,"region":region,
            "gpu_hour_usd":str(q.price_usd_per_hour),"disk_hour_allowance_usd":str(DISK_ALLOWANCE),
            "all_in_hour_ceiling_usd":str(total),"maximum_wall_seconds":3600,
            "existing_volume":GLOBAL,"disk_price_source":"https://docs.runpod.io/pods/pricing",
            "observed_at":datetime.now(timezone.utc).isoformat(),"allocation_attempted":False}
