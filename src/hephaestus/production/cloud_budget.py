"""Cloud-independent, evidence-bound autonomous compute budget policy.

An explicit owner-selected policy may authorize full existing *prepaid* balance;
unknown balances and extra top-ups are never inferred permissions. An expired
15-minute pause may resume ONLY within that original ceiling, at a freshly
validated price/balance with preserved evidence. No model self-authorizes
higher spending, no debt and no indefinite charge for stalled pods.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

from hephaestus.providers.runpod.cost_intelligence import usd


class BudgetMode(str,Enum):
    FIXED="fixed_prepaid_limit"
    AVAILABLE="available_prepaid_balance_opt_in"


class Decision(str,Enum):
    RUN="run"
    PAUSE="pause_notify"
    RESUME="resume_within_existing_approval"
    STOP="stop_no_available_prepaid_funds"


@dataclass(frozen=True)
class OwnerBudget:
    """Requires positive owner consent, never chosen by a model."""
    mode: BudgetMode
    owner_approved: bool
    quote_scope_id: str
    fixed_ceiling_usd: Decimal | None = None
    reserve_usd: Decimal = Decimal("0")
    no_response_pause_seconds: int = 900

    def validate(self):
        if not self.owner_approved or not self.quote_scope_id:
            raise PermissionError("missing explicit compute budget delegation")
        if self.no_response_pause_seconds!=900:
            raise ValueError("only the reviewed 15-minute fallback is supported")
        if self.mode==BudgetMode.FIXED:
            if self.fixed_ceiling_usd is None or usd(self.fixed_ceiling_usd)<=0:
                raise PermissionError("user fixed monetary budget is missing")
        elif self.mode==BudgetMode.AVAILABLE:
            if self.fixed_ceiling_usd is not None:
                raise PermissionError("account-balance mode must not invent a fixed budget")
        else:
            raise PermissionError("unknown budget mode")
        usd(self.reserve_usd)
        return self


@dataclass(frozen=True)
class BudgetEnvelope:
    """Immutable launch snapshot; cannot expand when account receives a top-up."""
    authorized_usd: Decimal
    account_balance_at_approval_usd: Decimal
    reserved_for_other_work_usd: Decimal
    reserved_for_storage_usd: Decimal
    quote_scope_id: str
    approved_at: datetime


@dataclass(frozen=True)
class ComputeEstimate:
    """Quoted provider rate + benchmark-derived duration in hours."""
    gpu_price_usd_per_hour: Decimal
    estimated_gpu_hours: Decimal
    existing_account_spend_usd_per_hour: Decimal
    estimated_other_charges_usd: Decimal
    source_identity: str

    def total_usd(self):
        if not self.source_identity:
            raise PermissionError("unproven price quote or runtime source")
        price=usd(self.gpu_price_usd_per_hour)
        hours=usd(self.estimated_gpu_hours)
        other=usd(self.estimated_other_charges_usd)
        existing=usd(self.existing_account_spend_usd_per_hour)
        return price*hours+existing*hours+other


def freeze_owner_budget(owner:OwnerBudget, *, live_prepaid_balance,
                        active_commitments_usd, storage_reserve_usd,
                        now:datetime) -> BudgetEnvelope:
    owner.validate()
    balance=usd(live_prepaid_balance)
    commitments=usd(active_commitments_usd)
    storage=usd(storage_reserve_usd)+usd(owner.reserve_usd)
    available=max(Decimal("0"),balance-commitments-storage)
    ceiling=available
    if owner.mode==BudgetMode.FIXED:
        ceiling=min(available,usd(owner.fixed_ceiling_usd))
    if ceiling<=0:
        raise PermissionError("no verified remaining prepaid cloud budget after commitments/reserve")
    if now.tzinfo is None:
        raise ValueError("budget snapshots need aware UTC timestamps")
    return BudgetEnvelope(ceiling,balance,commitments,storage,
                          owner.quote_scope_id,now)


def budget_decision(envelope:BudgetEnvelope, *, observed_spent_usd,
                    newly_quoted_cost_usd, verified_current_balance_usd,
                    live_existing_reservations_usd, now:datetime,
                    pause_started_at:datetime|None=None,
                    user_rejected:bool=False,
                    interruption_or_drift:bool=False):
    """Return decision and remaining ceiling. No timers or provider side-effects.

    Cross-budget / no-balance never has an auto-resume branch. On estimated
    cost increases, notify/pause 15 minutes. Silence then permits only work
    already delegated inside the unchanged snapshot, *not a larger grant*.
    """
    if now.tzinfo is None:
        raise ValueError("cloud decisions require aware wallclock")
    spent=usd(observed_spent_usd)
    ask=usd(newly_quoted_cost_usd)
    balance=usd(verified_current_balance_usd)
    commitments=usd(live_existing_reservations_usd)
    remaining=min(
        max(Decimal("0"),envelope.authorized_usd-spent),
        max(Decimal("0"),balance-commitments-envelope.reserved_for_storage_usd),
    )
    if user_rejected or ask>remaining or remaining<=0:
        return Decision.STOP, remaining
    if interruption_or_drift:
        if pause_started_at is None:
            return Decision.PAUSE, remaining
        if pause_started_at.tzinfo is None or pause_started_at>now:
            raise ValueError("invalid price-drift pause timestamp")
        if now-pause_started_at<timedelta(seconds=900):
            return Decision.PAUSE, remaining
        return Decision.RESUME, remaining
    return Decision.RUN,remaining


def pricing_notification(estimate:ComputeEstimate, envelope:BudgetEnvelope, *,
                         price_version:str, remaining_now):
    if not price_version:
        raise ValueError("a sourced dated quote is required")
    total=estimate.total_usd()
    return {
      "schema":"owner-approved-cloud-cost-projection.v1",
      "quote_scope_id":envelope.quote_scope_id,
      "estimate_usd":str(total),
      "initial_authorized_ceiling_usd":str(envelope.authorized_usd),
      "remaining_verified_prepaid_usd":str(usd(remaining_now)),
      "price_provenance":price_version,
      "estimated_gpu_hours":str(usd(estimate.estimated_gpu_hours)),
      "price_is_estimate_not_actual_bill":True,
      "overestimate_triggers_15minute_pause":True,
      "silence_may_resume_only_inside_unmodified_hard_envelope":True,
      "credit_topup_permission":False,
    }
