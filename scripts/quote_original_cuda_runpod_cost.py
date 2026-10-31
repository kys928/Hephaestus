#!/usr/bin/env python3
"""Read-only actual RunPod prepaid balance and GPU price; create cost quote.

Never launches pods, changes billing, or approves a GPU grant. The quote is a
separate input to the independently signed scientific compute authorization.
"""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import json,os
from pathlib import Path

from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing,usd
from hephaestus.production.cloud_budget import (
    BudgetMode,OwnerBudget,ComputeEstimate,freeze_owner_budget,pricing_notification
)
from hephaestus.storage.verified import canonical_bytes


def quote(args, *, provider, now):
    account=provider.account()
    if account.autopay_enabled:
        raise PermissionError("prepaid-only full-balance authorization requires auto-pay disabled")
    rate=provider.quote_gpu(gpu_id=args.gpu_id,
                            datacenter_id=args.datacenter_id,cloud=args.cloud)
    duration=usd(args.estimated_hours)
    if duration<=0:
        raise PermissionError("real benchmark-based GPU duration estimate is required")
    if args.budget_mode=="full-balance" and not args.owner_approved:
        raise PermissionError("full RunPod account balance must be owner opted in")
    policy=OwnerBudget(
        mode=BudgetMode.AVAILABLE if args.budget_mode=="full-balance" else BudgetMode.FIXED,
        owner_approved=args.owner_approved,
        quote_scope_id=args.campaign_id,
        fixed_ceiling_usd=None if args.budget_mode=="full-balance" else usd(args.fixed_budget_usd),
    )
    # Existing account charges are reserved independently; do not assume the
    # RunPod balance is isolated from other pods/serverless/storage.
    already_spending=account.already_spending_usd_per_hour*duration
    envelope=freeze_owner_budget(
        policy,live_prepaid_balance=account.credit_balance_usd,
        active_commitments_usd=already_spending,
        storage_reserve_usd=usd(args.storage_reserve_usd),now=now)
    estimate=ComputeEstimate(
        gpu_price_usd_per_hour=rate.price_usd_per_hour,
        estimated_gpu_hours=duration,
        existing_account_spend_usd_per_hour=account.already_spending_usd_per_hour,
        estimated_other_charges_usd=usd(args.storage_reserve_usd),
        source_identity="runpod_graphql_live_balance_and_rest_v2_catalog",
    )
    report=pricing_notification(estimate,envelope,price_version=now.isoformat(),
        remaining_now=account.credit_balance_usd)
    gpu_cost=rate.price_usd_per_hour*duration
    report.update({
        "provider":"RunPod","quote_status":"estimate_not_guaranteed",
        "gpu_id":rate.gpu_id,"cloud":rate.cloud,"gpu_memory_gb":rate.gpu_memory_gb,
        "datacenter_id":rate.datacenter_id,
        "quoted_gpu_hourly_usd":str(rate.price_usd_per_hour),
        "estimated_requested_gpu_compute_usd":str(gpu_cost),
        "preexisting_spend_rate_usd_per_hour":str(account.already_spending_usd_per_hour),
        "storage_reserved_usd":str(usd(args.storage_reserve_usd)),
        "prepaid_balance_snapshot_usd":str(account.credit_balance_usd),
        "quote_fits_available_authority":gpu_cost<=envelope.authorized_usd,
        "owner_approved_scope":args.owner_approved,
        "auto_pay_enabled":False,
        "explicit_compute_signature_required":True,
        "runpod_pod_created":False,
        "protected_historical_references_certified":False,
    })
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu-id",required=True)
    p.add_argument("--datacenter-id",required=True)
    p.add_argument("--cloud",choices=("SECURE","COMMUNITY"),default="SECURE")
    p.add_argument("--campaign-id",required=True)
    p.add_argument("--estimated-hours",required=True)
    p.add_argument("--storage-reserve-usd",required=True)
    p.add_argument("--budget-mode",choices=("full-balance","fixed"),default="fixed")
    p.add_argument("--fixed-budget-usd")
    p.add_argument("--owner-approved",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("original independent cost estimate must be immutable")
    key=os.environ.get("RUNPOD_READONLY_API_KEY","")
    result=quote(args,provider=ReadOnlyRunPodPricing(bearer_token=key),
                 now=datetime.now(timezone.utc))
    args.output.write_bytes(canonical_bytes(result))
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
