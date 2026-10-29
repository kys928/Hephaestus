"""Budget decisions are deterministic; these fixtures never contact RunPod."""
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import pytest

from hephaestus.production.cloud_budget import (
    BudgetMode,OwnerBudget,ComputeEstimate,Decision,
    freeze_owner_budget,budget_decision,pricing_notification,
)
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing


NOW=datetime(2026,10,29,8,tzinfo=timezone.utc)


def full():
    return OwnerBudget(BudgetMode.AVAILABLE,True,"campaign-102-gpu-original-five")


def decision(envelope, **extra):
    opts={"observed_spent_usd":"0","newly_quoted_cost_usd":"3",
          "verified_current_balance_usd":"20",
          "live_existing_reservations_usd":"2","now":NOW}
    opts.update(extra)
    return budget_decision(envelope,**opts)


def test_full_balance_requires_explicit_opt_in_and_reserves_private_storage():
    with pytest.raises(PermissionError):
        freeze_owner_budget(OwnerBudget(BudgetMode.AVAILABLE,False,"campaign"),
                            live_prepaid_balance="50",active_commitments_usd="3",
                            storage_reserve_usd="5",now=NOW)
    approved=freeze_owner_budget(full(),live_prepaid_balance="50",
                    active_commitments_usd="3",storage_reserve_usd="5",now=NOW)
    assert approved.authorized_usd==Decimal("42")
    assert approved.reserved_for_storage_usd==Decimal("5")
    assert approved.account_balance_at_approval_usd==Decimal("50")


def test_fixed_limit_preferred_and_cannot_spend_beyond_prepaid_balance():
    fixed=OwnerBudget(BudgetMode.FIXED,True,"run",Decimal("7"))
    env=freeze_owner_budget(fixed,live_prepaid_balance="100",
        active_commitments_usd="10",storage_reserve_usd="2",now=NOW)
    assert env.authorized_usd==Decimal("7")
    almost=freeze_owner_budget(fixed,live_prepaid_balance="5",
        active_commitments_usd="1",storage_reserve_usd="2",now=NOW)
    assert almost.authorized_usd==Decimal("2")


def test_15_minute_pauses_price_drift_then_resumes_within_original_ceiling():
    env=freeze_owner_budget(full(),live_prepaid_balance="20",
        active_commitments_usd="2",storage_reserve_usd="1",now=NOW)
    assert decision(env,interruption_or_drift=True)[0]==Decision.PAUSE
    assert decision(env,interruption_or_drift=True,
                    pause_started_at=NOW-timedelta(minutes=14))[0]==Decision.PAUSE
    result,unused=decision(env,interruption_or_drift=True,
                    pause_started_at=NOW-timedelta(minutes=15))
    assert result==Decision.RESUME
    assert unused==Decimal("17")


@pytest.mark.parametrize("cost,balance,spent,expected",[
    ("99","20","0",Decision.STOP),
    ("10","1","0",Decision.STOP),
    ("10","20","16",Decision.STOP),
    ("1","20","0",Decision.RUN),
])
def test_no_silence_can_authorize_spending_beyond_hard_ceiling(cost,balance,spent,expected):
    env=freeze_owner_budget(full(),live_prepaid_balance="20",
        active_commitments_usd="1",storage_reserve_usd="1",now=NOW)
    outcome,_=decision(env,newly_quoted_cost_usd=cost,
        verified_current_balance_usd=balance,observed_spent_usd=spent,
        pause_started_at=NOW-timedelta(hours=2),interruption_or_drift=True)
    assert outcome==(Decision.RESUME if expected==Decision.RUN else expected)


def test_new_topup_cannot_enlarge_approved_budget_even_after_15min():
    env=freeze_owner_budget(full(),live_prepaid_balance="10",
        active_commitments_usd="2",storage_reserve_usd="1",now=NOW)
    status,remaining=decision(env,newly_quoted_cost_usd="8",
        verified_current_balance_usd="200",
        pause_started_at=NOW-timedelta(hours=2),interruption_or_drift=True)
    assert status==Decision.STOP
    assert remaining==Decimal("7")


def test_dynamic_quote_includes_all_expected_compute_and_overhead():
    env=freeze_owner_budget(full(),live_prepaid_balance="40",
        active_commitments_usd="0",storage_reserve_usd="3",now=NOW)
    quote=ComputeEstimate(Decimal("0.80"),Decimal("6"),
        Decimal("0.25"),Decimal("2"),"RunPod v2 2026-10-29 catalogue")
    assert quote.total_usd()==Decimal("8.30")
    report=pricing_notification(quote,env,price_version="published-live-price",remaining_now="40")
    assert report["estimate_usd"]=="8.30"
    assert report["credit_topup_permission"] is False


def test_read_only_live_runpod_balance_requires_auth_and_valid_json():
    responses={
        "https://api.runpod.io/graphql":{
            "data":{"myself":{"clientBalance":14.6,
                "currentSpendPerHr":0.12,"isAutoPayEnabled":False}}
        },
        "https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD":{
            "gpus":[{"id":"NVIDIA GeForce RTX 4090","name":"RTX 4090",
                    "memory":24,"secure":True,"community":True,
                    "price":{"secure":0.55,"community":0.34}}]}
    }
    def fake(url,method,data):
        assert method==("POST" if url.endswith("graphql") else "GET")
        return responses[url]
    client=ReadOnlyRunPodPricing(bearer_token="TEST_ONLY_NOT_SECRET",transport=fake)
    observed=client.account()
    assert observed.credit_balance_usd==Decimal("14.6")
    assert observed.autopay_enabled is False
    price=client.quote_gpu(gpu_id="NVIDIA GeForce RTX 4090",datacenter_id="EU-RO-1")
    assert price.price_usd_per_hour==Decimal("0.55")
    with pytest.raises(RuntimeError):
        client.quote_gpu(gpu_id="invented-gpu",datacenter_id="EU-RO-1")
    responses["https://api.runpod.io/graphql"]={"errors":[{"message":"rate limited"}]}
    with pytest.raises(RuntimeError):
        client.account()


@pytest.mark.parametrize("input",["NaN","Infinity","-0.01",None,True])
def test_invalid_provider_balance_always_fails_closed(input):
    from hephaestus.providers.runpod.cost_intelligence import usd
    with pytest.raises(ValueError):
        usd(input)
