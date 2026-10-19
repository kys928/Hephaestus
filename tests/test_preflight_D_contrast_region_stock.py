"""No-network tests of exact in-datacenter stock and price gates."""
import pytest
import preflight_D_contrast_region_stock as stock

def packet(status="High",price=2.49,counts=(1,2)):
    return {"data":{"gpuTypes":[{"id":stock.GPU,"lowestPrice":{
        "stockStatus":status,"uninterruptablePrice":price,
        "availableGpuCounts":list(counts)}}]}}

@pytest.mark.parametrize("status,price,counts,allowed",[
    ("High",2.49,(1,2),True),
    ("Medium",2.40,(1,),True),
    ("Low",2.49,(1,),False),
    ("None",2.49,(),False),
    ("High",2.50,(1,),False),
    ("High",2.49,(2,),False),
])
def test_exact_gpu_datacenter_price_and_capacity_threshold(status,price,counts,allowed):
    assert stock.inspect_response(packet(status,price,counts))["confident_capacity_signal"] is allowed

@pytest.mark.parametrize("payload",[
    {},
    {"errors":[{"message":"denied"}]},
    {"data":{"gpuTypes":[]}},
    {"data":{"gpuTypes":[{"id":"NVIDIA RTX 4090","lowestPrice":{}}]}},
    {"data":{"gpuTypes":[{"id":"NVIDIA RTX PRO 6000 Blackwell Server Edition","lowestPrice":{"stockStatus":"Maybe","uninterruptablePrice":2.49,"availableGpuCounts":[1]}}]}},
])
def test_unknown_or_untrusted_provider_signal_fails_closed(payload):
    with pytest.raises(ValueError):stock.inspect_response(payload)

def test_price_or_count_schema_must_be_numeric():
    with pytest.raises(ValueError):stock.inspect_response(packet(price="2.49"))
    with pytest.raises(ValueError):stock.inspect_response(packet(counts=(True,)))

def test_constrains_request_to_network_volume_location(monkeypatch):
    class Dummy:
        def __init__(self,req,timeout):
            import json
            envelope=json.loads(req.data.decode())
            assert set(envelope)=={"query"}
            query=envelope["query"]
            assert 'dataCenterId: "EU-CZ-1"' in query
            assert "secureCloud: true" in query
            assert "gpuCount: 1" in query
            assert 'id: "NVIDIA RTX PRO 6000 Blackwell Server Edition"' in query
            self.closed=False
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def read(self):return b'{"data":{"gpuTypes":[]}}'
    monkeypatch.setenv("RUNPOD_API_KEY","dummy-never-logged")
    monkeypatch.setattr(stock.urllib.request,"urlopen",lambda req,timeout:Dummy(req,timeout))
    assert stock.query()["data"]["gpuTypes"]==[]

def test_low_stock_gate_does_not_trigger_gpu_request(monkeypatch):
    monkeypatch.setattr(stock,"query",lambda:packet("Low"))
    assert stock.main()==75
    monkeypatch.setattr(stock,"query",lambda:packet("High"))
    assert stock.main()==0
