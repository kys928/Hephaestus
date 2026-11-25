"""Read authenticated Secure Cloud catalog and candidate placement, without allocation."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os

from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing


def main():
    resource = ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    candidate, region = "NVIDIA L40S", "EU-CZ-1"
    quote = resource.quote_gpu(gpu_id=candidate, datacenter_id=region)
    catalog = resource._request(
        f"https://api.runpod.io/v2/catalog/datacenters/{region}?include=GPU_AVAILABILITY", "GET", None,
    )
    if catalog.get("id") != region:
        raise ValueError("provider returned a different placement")
    stock = next((row.get("availability") for row in catalog.get("gpuAvailability", [])
                  if row.get("id") == candidate), None)
    account = resource.account()
    print(json.dumps({
        "quote": asdict(quote),
        "placement_availability": stock,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_hour_gpu_only_funded": account.credit_balance_usd >= quote.price_usd_per_hour,
        "allocation_authorized": False,
    }, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
