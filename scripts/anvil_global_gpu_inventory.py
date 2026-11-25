"""Inspect authenticated regional stock for the existing Global Volume."""
from decimal import Decimal
import json
import os

from hephaestus.providers.runpod.cost_intelligence import REST_URL, ReadOnlyRunPodPricing, usd

API = "https://api.runpod.io/v2"
GLOBAL = "cmw51pjsk000007l9a0mv1jga"
REGIONS = ("EU-CZ-1", "EU-RO-1", "EU-SE-1", "EU-NL-1", "EU-FR-1", "EU-DE-1", "EU-FI-1", "EU-NO-1")


def main():
    client = ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    volume = client._request(f"{API}/global-volumes/{GLOBAL}", "GET", None)
    if volume.get("id") != GLOBAL:
        raise PermissionError("Global Volume identity differs")
    gpu_catalog = client._request(REST_URL, "GET", None)["gpus"]
    usable = {row["id"]: row for row in gpu_catalog if row.get("secure") is True
              and type(row.get("memory")) is int and row["memory"] >= 48 and row.get("price", {}).get("secure")}
    candidates = []
    for region in REGIONS:
        try:
            catalog = client._request(f"{API}/catalog/datacenters/{region}?include=GPU_AVAILABILITY", "GET", None)
        except Exception as error:
            print(json.dumps({"region": region, "catalog_status": getattr(error, "code", None),
                              "error_type": type(error).__name__}), flush=True)
            continue
        if catalog.get("id") != region:
            raise PermissionError("wrong regional catalog")
        for row in catalog.get("gpuAvailability", []):
            gpu = usable.get(row.get("id"))
            if gpu and row.get("availability") in {"LOW", "MEDIUM", "HIGH"}:
                candidates.append({"region": region, "gpu_id": gpu["id"], "vram_gb": gpu["memory"],
                                   "secure_gpu_hour_usd": str(usd(gpu["price"]["secure"])),
                                   "availability": row["availability"]})
    candidates.sort(key=lambda x: (Decimal(x["secure_gpu_hour_usd"]), x["region"], x["gpu_id"]))
    account = client.account()
    print(json.dumps({"global_volume_confirmed": True, "candidates": candidates[:12],
                      "gpu_only_funded": bool(candidates and account.credit_balance_usd >=
                                              Decimal(candidates[0]["secure_gpu_hour_usd"])),
                      "allocation_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
