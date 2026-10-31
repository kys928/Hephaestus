"""Read-only RunPod price and spend observations; never invent an account balance.

RunPod REST v2 supplies pod GPU catalog prices. Live prepaid clientBalance
still comes from the documented GraphQL myself query. Both results are
untrusted until transport/authentication succeeds; no stale-price spending.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable, Mapping
from urllib.request import Request, urlopen
import json

REST_URL="https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD"
BALANCE_URL="https://api.runpod.io/graphql"
_BALANCE_QUERY="query HephaestusReadOnlyBudget { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"


def usd(value):
    if isinstance(value,bool) or value is None:
        raise ValueError("unknown provider USD value")
    try:
        number=Decimal(str(value))
    except (InvalidOperation,ValueError,TypeError) as exc:
        raise ValueError("invalid USD amount") from exc
    if not number.is_finite() or number<0:
        raise ValueError("provider USD amount must be finite nonnegative")
    return number


@dataclass(frozen=True)
class RunPodAccountObservation:
    credit_balance_usd: Decimal
    already_spending_usd_per_hour: Decimal
    autopay_enabled: bool
    source: str = "runpod_authenticated_myself.clientBalance"


@dataclass(frozen=True)
class RunPodGpuQuote:
    gpu_id: str
    cloud: str
    gpu_memory_gb: int
    price_usd_per_hour: Decimal
    datacenter_id: str
    source: str = "runpod_v2_catalog_gpus_POD"


class ReadOnlyRunPodPricing:
    def __init__(self, *, bearer_token:str, transport:Callable|None=None):
        if not isinstance(bearer_token,str) or not bearer_token.strip():
            raise PermissionError("RunPod read-only authenticated pricing requires credentials")
        self._token=bearer_token
        self._transport=transport or self._request

    def _request(self, url, method, data):
        encoded=json.dumps(data).encode() if data is not None else None
        req=Request(url,method=method,data=encoded,headers={
            "Authorization":"Bearer "+self._token,"Content-Type":"application/json"})
        with urlopen(req,timeout=15) as response:
            if response.status!=200:
                raise RuntimeError("live RunPod price/balance request unavailable")
            return json.load(response)

    def account(self):
        raw=self._transport(BALANCE_URL,"POST",{"query":_BALANCE_QUERY})
        if (not isinstance(raw,dict) or raw.get("errors")
                or not isinstance(raw.get("data"),dict)
                or not isinstance(raw["data"].get("myself"),dict)):
            raise RuntimeError("no authenticated live RunPod balance; refuse unbounded spending")
        user=raw["data"]["myself"]
        if type(user.get("isAutoPayEnabled")) is not bool:
            raise RuntimeError("unknown RunPod autopay status")
        return RunPodAccountObservation(
            credit_balance_usd=usd(user["clientBalance"]),
            already_spending_usd_per_hour=usd(user["currentSpendPerHr"]),
            autopay_enabled=user["isAutoPayEnabled"],
        )

    def quote_gpu(self, *, gpu_id:str, datacenter_id:str, cloud="SECURE"):
        if cloud not in ("SECURE","COMMUNITY") or not datacenter_id or not gpu_id:
            raise PermissionError("quote requires exact GPU, datacenter and cloud tier")
        raw=self._transport(REST_URL,"GET",None)
        if not isinstance(raw,dict) or not isinstance(raw.get("gpus"),list):
            raise RuntimeError("live RunPod REST-v2 GPU price catalog unavailable")
        matches=[item for item in raw["gpus"]
                 if isinstance(item,dict) and item.get("id")==gpu_id]
        if len(matches)!=1:
            raise RuntimeError("exact RunPod GPU ID unavailable in live catalog")
        item=matches[0]
        tier=cloud.lower()
        if item.get(tier) is not True:
            raise RuntimeError("requested cloud tier not offered for that GPU")
        prices=item.get("price")
        if not isinstance(prices,dict):
            raise RuntimeError("no authoritative GPU price in current catalog")
        price=usd(prices.get(tier))
        if price<=0:
            raise RuntimeError("GPU provider price must be positive")
        # Datacenter stock is separately checked by the allocation adapter's
        # live catalog/datacenters GPU_AVAILABILITY query.
        memory=item.get("memory")
        if type(memory) is not int or memory<=0:
            raise RuntimeError("GPU VRAM unavailable; cannot choose safe placement")
        return RunPodGpuQuote(gpu_id,cloud,memory,price,datacenter_id)
