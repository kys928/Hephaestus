from decimal import Decimal
from types import SimpleNamespace
from hephaestus.providers.runpod.anvil_dynamic_placement import discover
from hephaestus.providers.runpod.anvil_live_lease import GLOBAL
from hephaestus.providers.runpod.cost_intelligence import REST_URL

class Pricing:
    def account(self):
        return SimpleNamespace(autopay_enabled=False,credit_balance_usd=Decimal("5"),already_spending_usd_per_hour=Decimal("0"))
    def _request(self,url,*args):
        if "global-volumes" in url:return {"id":GLOBAL}
        if url==REST_URL:return {"gpus":[{"id":gpu,"secure":True,"memory":memory,"price":{"secure":price}}
            for gpu,memory,price in [("NVIDIA A40",48,".59"),("NVIDIA RTX A6000",48,".55"),("NVIDIA L40S",48,".99"),("unreviewed",80,".1")]]}
        if url.endswith("datacenters"):return [{"id":"CA-MTL-1"},{"id":"EU-SE-1"}]
        region=url.split("/")[-1].split("?")[0]
        return {"id":region,"gpuAvailability":[{"id":sku,"availability":"LOW"} for sku in ["NVIDIA A40","NVIDIA RTX A6000","NVIDIA L40S","unreviewed"]]}

def test_dynamic_selection_filters_and_ranks():
    rows=discover(Pricing())
    assert len(rows)==4
    assert rows[0]["gpu"]=="NVIDIA RTX A6000" and rows[0]["region"]=="CA-MTL-1"
    assert all(Decimal(x["all_in_hour_ceiling_usd"])<=Decimal(".65") for x in rows)

def test_discovery_rechecks_missing_then_available():
    class Flaky(Pricing):
        calls=0
        def _request(self,url,*args):
            if "GPU_AVAILABILITY" in url:
                self.calls+=1
                if self.calls<=2:
                    return {"id":url.split("/")[-1].split("?")[0]}
            return super()._request(url,*args)
    docs=[]; pauses=[]
    rows=discover(Flaky(),docs.append,pause=pauses.append)
    assert rows and pauses==[2]
    assert [d["scan_attempt"] for d in docs if "scan_attempt" in d]==[1,2]
    assert next(d for d in docs if "unverified" in d)["unverified"][0]["reason"]=="availability_missing_or_invalid"


def test_discovery_stops_after_three_empty_scans():
    class Empty(Pricing):
        def _request(self,url,*args):
            if "GPU_AVAILABILITY" in url:
                return {"id":url.split("/")[-1].split("?")[0]}
            return super()._request(url,*args)
    docs=[]; pauses=[]
    assert discover(Empty(),docs.append,pause=pauses.append)==[]
    assert len([d for d in docs if "scan_attempt" in d])==3 and pauses==[2,4]


def test_shorter_lease_qualifies_realistic_balance_without_weakening_ceiling():
    from hephaestus.providers.runpod.anvil_live_lease import fundable,account_draw
    a=SimpleNamespace(autopay_enabled=False,credit_balance_usd=Decimal("1.21"),already_spending_usd_per_hour=Decimal(".797"))
    assert fundable(a,Decimal(".60"))
    assert not fundable(a,Decimal(".60"),3600)
    assert not fundable(a,Decimal(".70"))
    a.credit_balance_usd=Decimal(".90")
    assert not fundable(a,Decimal(".60"))
