from decimal import Decimal
from types import SimpleNamespace
import json
import pytest
from hephaestus.providers.runpod.anvil_placement_diagnostics import inspect_placement, PlacementFailure, safe_failure
from hephaestus.providers.runpod.anvil_live_lease import GPU,GLOBAL,REGION
from hephaestus.providers.runpod.execution import RunPodApiError,AmbiguousRunPodOperation

class Pricing:
    def quote_gpu(self,**kw):
        return SimpleNamespace(price_usd_per_hour=Decimal(".59"),gpu_memory_gb=48)
    def account(self):
        return SimpleNamespace(autopay_enabled=False,credit_balance_usd=Decimal("3"),already_spending_usd_per_hour=Decimal("0"))

class Provider:
    stock="LOW"
    def _request(self,method,path):
        if path.startswith("catalog/"):
            return 200,{"id":REGION,"gpuAvailability":[{"id":GPU,"availability":self.stock}] if self.stock else []}
        return 200,{"id":GLOBAL}
    def select_gpu(self,allowed):
        return GPU

def test_placement_success_separate_stages():
    docs=[]
    quote=inspect_placement(Pricing(),Provider(),region=REGION,report=docs.append)
    assert quote["all_in_hour_ceiling_usd"]=="0.60"
    assert [x["stage"] for x in docs]==["gpu_quote","regional_catalog","global_volume","prepaid_budget","final_stock_recheck"]
    assert all(not x["allocation_attempted"] for x in docs)

@pytest.mark.parametrize("stock,reason",[(None,"no_allowed_stock"),("NONE","no_allowed_stock"),("UNEXPECTED_SECRET","unknown_stock_status")])
def test_missing_or_unknown_stock(stock,reason):
    p=Provider()
    p.stock=stock
    docs=[]
    with pytest.raises(PlacementFailure) as err:
        inspect_placement(Pricing(),p,region=REGION,report=docs.append)
    assert err.value.diagnostic["reason"]==reason
    assert "UNEXPECTED_SECRET" not in json.dumps(docs)

def test_stock_lost_at_final_gate():
    p=Provider()
    def lost(_):
        raise RunPodApiError("No allowed GPU available in the pinned datacenter; no allocation attempted")
    p.select_gpu=lost
    with pytest.raises(PlacementFailure) as err:
        inspect_placement(Pricing(),p,region=REGION,report=lambda _:None)
    assert err.value.diagnostic["stage"]=="final_stock_recheck"
    assert err.value.diagnostic["reason"]=="no_allowed_stock"

@pytest.mark.parametrize("code,reason",[(401,"authentication_or_permission"),(403,"authentication_or_permission"),(429,"rate_limit"),(503,"provider_server_error")])
def test_safe_provider_error(code,reason):
    doc=safe_failure("regional_catalog",RunPodApiError("private raw response SECRET",status_code=code))
    assert doc["reason"]==reason and doc["http_status"]==code
    assert "SECRET" not in json.dumps(doc)

def test_ambiguous_post_is_not_stock_failure():
    assert safe_failure("allocation",AmbiguousRunPodOperation("secret"))["reason"]=="allocation_outcome_unknown_reconcile_no_retry"


def test_placement_retains_actual_nondefault_gpu_identity():
    gpu="NVIDIA RTX A6000"
    class NonDefault(Provider):
        def _request(self,method,path):
            if path.startswith("catalog/"):
                return 200,{"id":REGION,"gpuAvailability":[{"id":gpu,"availability":"LOW"}]}
            return 200,{"id":GLOBAL}
        def select_gpu(self,allowed):
            assert allowed==[gpu]
            return gpu
    assert inspect_placement(Pricing(),NonDefault(),region=REGION,gpu=gpu,report=lambda _:None)["gpu"]==gpu

@pytest.mark.parametrize("payload,identity,present,kind",[
    ({"id":REGION},True,False,"NoneType"),
    ({"id":REGION,"gpuAvailability":None},True,True,"NoneType"),
    ({"id":"other","gpuAvailability":[]},False,True,"list"),
])
def test_catalog_schema_failure_reports_safe_shape(payload,identity,present,kind):
    class Invalid(Provider):
        def _request(self,method,path):
            return 200,payload
    with pytest.raises(PlacementFailure) as err:
        inspect_placement(Pricing(),Invalid(),region=REGION,report=lambda _:None)
    d=err.value.diagnostic
    assert (d["identity_matches"],d["availability_present"],d["availability_type"])==(identity,present,kind)
