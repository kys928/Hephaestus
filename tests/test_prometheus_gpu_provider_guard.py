"""Offline negative paths for the one-time RunPod supervisor pair.

No provider network calls or paid Pod creations take place in these tests.
"""
from __future__ import annotations

from datetime import datetime,timedelta,timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hephaestus.development.prometheus_gpu_provider import (
    _wallet_lease, _pod_command, _source_scope, _require_independent_watchdog,
    MAX_LEASE_SECONDS, _s3_existing, GPU_PREFERENCES, _require_admitted_intent_root,
)
from hephaestus.development.prometheus_gpu_watchdog import _matching
from hephaestus.storage.verified import canonical_bytes


def test_source_scoped_sqlite_intent_root_is_admitted_local_tmp_only():
    sha = "a" * 40
    _require_admitted_intent_root(Path("/tmp/prometheus-runpod-" + sha), sha)
    with pytest.raises(PermissionError, match="not admitted"):
        _require_admitted_intent_root(Path("/home/runner/work/_temp/prometheus-operator-state"), sha)
    with pytest.raises(PermissionError, match="not admitted"):
        _require_admitted_intent_root(Path("/tmp/prometheus-runpod-" + "b" * 40), sha)


def test_one_time_gpu_is_pinned_to_exact_operator_selected_model():
    # No silent GPU-family substitution on the fresh paid authorization.
    assert GPU_PREFERENCES == ("NVIDIA RTX PRO 6000 Blackwell Server Edition",)


def test_prepaid_finite_execution_is_not_an_unbounded_money_sink():
    account=SimpleNamespace(autopay_enabled=False,
        credit_balance_usd=Decimal("18.0"),
        already_spending_usd_per_hour=Decimal("0"))
    quote=SimpleNamespace(price_usd_per_hour=Decimal("2.0"))
    seconds=_wallet_lease(account,quote)
    assert 900<=seconds<=MAX_LEASE_SECONDS
    assert seconds==MAX_LEASE_SECONDS


def test_prepaid_expiry_blocks_without_new_credit():
    account=SimpleNamespace(autopay_enabled=False,
        credit_balance_usd=Decimal("0.40"),
        already_spending_usd_per_hour=Decimal("0"))
    quote=SimpleNamespace(price_usd_per_hour=Decimal("2.5"))
    with pytest.raises(PermissionError,match="no available"):
        _wallet_lease(account,quote)


def test_autopay_debt_is_not_implicit_unlimited_authority():
    account=SimpleNamespace(autopay_enabled=True,
        credit_balance_usd=Decimal("100"),
        already_spending_usd_per_hour=Decimal("0"))
    quote=SimpleNamespace(price_usd_per_hour=Decimal("2"))
    with pytest.raises(PermissionError,match="autopay"):
        _wallet_lease(account,quote)


def test_pod_command_uses_exact_code_hash_and_disallows_out_of_scope_duration(monkeypatch):
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY_ID","fake-test-only")
    monkeypatch.setenv("RUNPOD_S3_SECRET_ACCESS_KEY","fake-test-only")
    sha="a"*64
    cmd,env=_pod_command("hephaestus/scientific/prometheus/validated",sha,1800)
    assert cmd[:2]==["bash","-lc"]
    assert "sha256sum" in cmd[2]
    assert "--out" in cmd[2]
    assert "--parent-report" in cmd[2]
    assert "--d-report" in cmd[2]
    assert "--s3-region EU-CZ-1" in cmd[2]
    assert "runpod/pytorch" not in cmd[2]
    assert env["RUNPOD_S3_ACCESS_KEY_ID"]=="fake-test-only"
    assert "RUNPOD_API_KEY" not in env
    with pytest.raises(ValueError):
        _pod_command("safe",sha,MAX_LEASE_SECONDS+1)
    with pytest.raises(ValueError):
        _pod_command("safe","bad",1200)


def test_source_scope_refuses_nonimmutable_commit():
    scope,name=_source_scope("a"*40)
    assert name.startswith("prometheus-")
    assert scope.endswith("a"*40)
    with pytest.raises(ValueError):
        _source_scope("branch-main")


class _Body:
    def __init__(self,data): self.data=io.BytesIO(data)
    def read(self,n=-1):return self.data.read(n)
    def close(self): self.data.close()


class _S3:
    def __init__(self,source=None):self.source=source
    def get_object(self,*,Bucket,Key):
        if self.source is None:
            e=RuntimeError("missing")
            e.response={"Error":{"Code":"NoSuchKey"}}
            raise e
        return {"Body":_Body(self.source)}


def test_launch_refuses_paid_allocation_without_independent_watchdog(monkeypatch):
    import hephaestus.development.prometheus_gpu_provider as l
    monkeypatch.setattr(l.time,"sleep",lambda sec: (_ for _ in ()).throw(RuntimeError("wait")))
    with pytest.raises(RuntimeError,match="wait"):
        _require_independent_watchdog(_S3(), "cviwpryzao",
                                      "hephaestus/scientific/prometheus/dummy", "a"*40)


def test_wrong_ready_receipt_is_denied_without_paid_allocation():
    ready={
        "schema":"prometheus-independent-offpod-watchdog.v1",
        "source_sha":"b"*40,
        "prefix":"hephaestus/scientific/prometheus/dummy",
        "network_volume_id":"cviwpryzao",
        "owner":"separately_scheduled_github_actions_watchdog",
        "hard_deadline":(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
    }
    with pytest.raises(PermissionError,match="identity mismatch"):
        _require_independent_watchdog(_S3(json.dumps(ready).encode()),
                                      "cviwpryzao",ready["prefix"],"a"*40)


def test_valid_ready_receipt_is_bound_to_campaign():
    sha="a"*40
    prefix,_=_source_scope(sha)
    ready={
        "schema":"prometheus-independent-offpod-watchdog.v1",
        "source_sha":sha,"prefix":prefix,
        "network_volume_id":"cviwpryzao",
        "owner":"separately_scheduled_github_actions_watchdog",
        "hard_deadline":(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
    }
    assert _require_independent_watchdog(_S3(json.dumps(ready).encode()),
                                         "cviwpryzao",prefix,sha)==ready


def test_watchdog_filters_to_immutable_source_and_volumes():
    sha="a"*40
    prefix,name=_source_scope(sha)
    config=SimpleNamespace(api_base_url="https://api.runpod.io/v2",
        datacenter_id="EU-CZ-1",network_volume_id="cviwpryzao")
    from hephaestus.development.prometheus_gpu_provider import IMAGE, ID
    operation_key=hashlib.sha256(canonical_bytes([
        config.api_base_url, config.network_volume_id, ID+"-"+sha,
    ])).hexdigest()
    good={
        "id":"paid-diagnostic-test",
        "name":name+"-"+operation_key[:32],
        "cloud":"SECURE",
        "dataCenterId":"EU-CZ-1","image":IMAGE,
        "gpu":{"id":"NVIDIA H100 80GB HBM3","count":1},
        "env":{"PROMETHEUS_CAMPAIGN_ID":prefix},
        "mounts":{"network":[{"volumeId":"cviwpryzao","path":"/workspace"}]},
    }
    adapter=SimpleNamespace(list_pods=lambda:[{"id":"unrelated","name":"other"},good])
    assert _matching(adapter,prefix=prefix,basename=name,config=config)==[good]
    other=dict(good,env={"PROMETHEUS_CAMPAIGN_ID":"different"})
    with pytest.raises(PermissionError):
        _matching(SimpleNamespace(list_pods=lambda:[other]),
                  prefix=prefix,basename=name,config=config)
    # Even matching GPU, mounts and env do not authorize a different Pod:
    # the provider operation ID must match the pinned campaign source.
    mimicked=dict(good,name=name+"-"+"f"*32)
    with pytest.raises(PermissionError, match="unexpected immutable"):
        _matching(SimpleNamespace(list_pods=lambda:[mimicked]),
                  prefix=prefix,basename=name,config=config)
    # Provider responses can omit credentials; exact source-bound identity
    # remains available even if env is redacted.
    redacted=dict(good)
    redacted.pop("env")
    assert _matching(SimpleNamespace(list_pods=lambda:[redacted]),
                     prefix=prefix,basename=name,config=config)==[redacted]
