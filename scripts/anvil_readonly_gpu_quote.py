"""Read authenticated Secure Cloud catalog and candidate placement, without allocation."""
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import json
import os

from hephaestus.providers.runpod.cost_intelligence import REST_URL, ReadOnlyRunPodPricing, usd


def main():
    resource = ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    candidate, region = "NVIDIA L40S", "EU-CZ-1"
    quote = resource.quote_gpu(gpu_id=candidate, datacenter_id=region)
    catalog = resource._request(
        f"https://api.runpod.io/v2/catalog/datacenters/{region}?include=GPU_AVAILABILITY", "GET", None,
    )
    if catalog.get("id") != region:
        raise ValueError("provider returned a different placement")
    stock_by_id = {
        row.get("id"): row.get("availability") for row in catalog.get("gpuAvailability", [])
        if isinstance(row, dict)
    }
    stock = stock_by_id.get(candidate)
    gpu_catalog = resource._request(REST_URL, "GET", None)
    alternatives = []
    for item in gpu_catalog.get("gpus", []):
        gpu_id, memory = item.get("id"), item.get("memory")
        if (stock_by_id.get(gpu_id) not in {"LOW", "MEDIUM", "HIGH"}
                or item.get("secure") is not True or type(memory) is not int or memory < 48):
            continue
        price = usd(item.get("price", {}).get("secure"))
        if price > 0:
            alternatives.append({"gpu_id": gpu_id, "memory_gb": memory,
                                 "price_usd_per_hour": str(price), "stock": stock_by_id[gpu_id]})
    alternatives.sort(key=lambda row: (Decimal(row["price_usd_per_hour"]), row["gpu_id"]))
    account = resource.account()
    print(json.dumps({
        "quote": asdict(quote),
        "placement_availability": stock,
        "same_region_secure_available_candidates": alternatives[:5],
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_hour_gpu_only_funded": account.credit_balance_usd >= quote.price_usd_per_hour,
        "allocation_authorized": False,
    }, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
