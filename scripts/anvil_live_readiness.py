"""Authenticate actual research scope and current placement without allocating."""
from decimal import Decimal
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from anvil_neon_bootstrap import connect_runtime, verify_runtime
from hephaestus.control.anvil_live import publish_frozen_inputs
from hephaestus.providers.runpod.anvil_live_lease import GLOBAL, REGION, GPU, MAX_RATE
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.state.anvil_postgres import PostgresAnvilState
from hephaestus.storage.filesystem import FileSystemArtifactStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    bundle = json.loads((ROOT / "configs/research/anvil_diagnosis_planner_first_live_v1.json").read_bytes())
    plan = bundle["plan"]["document"]
    with TemporaryDirectory(prefix="anvil-readiness-") as tmp:
        ref = publish_frozen_inputs(bundle, FileSystemArtifactStore(Path(tmp)))
    verify_runtime(plan["investigation"])
    board = PostgresAnvilState(connect_runtime, plan["investigation"], 1,
                               lambda _: False, lambda *_: False)
    state = board.replay(plan["investigation"], expected_frontier=bundle["expected_initial_frontier"])
    if state.contributions:
        raise PermissionError("live investigation already has contributions; never restart automatically")
    print(json.dumps({"schema": "anvil.live-readiness.v1", "plan_ref": ref,
                      "frontier": state.frontier, "accepted_count": 0,
                      "hosted_live_scope_verified": True}, sort_keys=True), flush=True)

    pricing = ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    quote = pricing.quote_gpu(gpu_id=GPU, datacenter_id=REGION)
    region = pricing._request(
        f"https://api.runpod.io/v2/catalog/datacenters/{REGION}?include=GPU_AVAILABILITY", "GET", None)
    global_volume = pricing._request(f"https://api.runpod.io/v2/global-volumes/{GLOBAL}", "GET", None)
    if region.get("id") != REGION or global_volume.get("id") != GLOBAL:
        raise PermissionError("placement or existing volume identity differs")
    stock = [row.get("availability") for row in region.get("gpuAvailability", []) if row.get("id") == GPU]
    account = pricing.account()
    print(json.dumps({"schema": "anvil.live-placement-observation.v1", "region": REGION,
        "gpu": GPU, "vram_gb": quote.gpu_memory_gb, "gpu_only_hourly_usd": str(quote.price_usd_per_hour),
        "stock": stock[0] if len(stock) == 1 else None, "existing_global_confirmed": True,
        "prepaid_ceiling_fundable": not account.autopay_enabled and
             account.credit_balance_usd > MAX_RATE + account.already_spending_usd_per_hour,
        "gpu_only_within_ceiling": Decimal(0) < quote.price_usd_per_hour < MAX_RATE,
        "all_in_quote_verified": False, "allocation_authorized": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit("Anvil readiness failed (" + type(error).__name__ + "); no allocation")
