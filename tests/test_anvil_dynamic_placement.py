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
