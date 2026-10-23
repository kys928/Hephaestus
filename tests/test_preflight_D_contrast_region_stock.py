"""No-network tests of exact in-datacenter stock and price gates."""
import pytest
import preflight_D_contrast_region_stock as stock

def packet(status="High",price=2.59,counts=(1,2)):
    return {"data":{"gpuTypes":[{"id":stock.GPU,"lowestPrice":{
        "stockStatus":status,"uninterruptablePrice":price,
        "availableGpuCounts":list(counts)}}]}}

@pytest.mark.parametrize("status,price,counts,allowed",[
    ("High",2.59,(1,2),True),
    ("Medium",2.50,(1,),True),
    ("Low",2.59,(1,),True),
    ("Low",2.60,(1,),False),
    ("None",2.59,(),False),
    ("High",2.60,(1,),False),
    ("High",2.59,(2,),False),
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

@pytest.mark.parametrize("status,allowed",[
    ("Low",True),("None",False),("Medium",True),("High",True)
])
def test_live_nullable_gpu_counts_are_not_a_schema_error(status,allowed):
    # RunPod GraphQL LowestPrice.availableGpuCounts is nullable. The
    # query requests GPU count 1 in EU-CZ-1, so use regional stock signal.
    quote=packet(status,2.59,(1,))
    quote["data"]["gpuTypes"][0]["lowestPrice"]["availableGpuCounts"]=None
    result=stock.inspect_response(quote)
    assert result["confident_capacity_signal"] is allowed
    assert result["available_gpu_counts"] is None
    if allowed:
        assert result["one_gpu_evidence"]=="regional_gpuCount_1_stock"
    else:
        assert result["availability_reason"]=="regional_stock_insufficient"


def test_explicit_count_list_without_one_stays_unavailable():
    result=stock.inspect_response(packet("High",2.59,(2,4)))
    assert result["confident_capacity_signal"] is False


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
            assert 'id: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"' in query
            self.closed=False
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def read(self):return b'{"data":{"gpuTypes":[]}}'
    monkeypatch.setenv("RUNPOD_API_KEY","dummy-never-logged")
    monkeypatch.setattr(stock.urllib.request,"urlopen",lambda req,timeout:Dummy(req,timeout))
    assert stock.query()["data"]["gpuTypes"]==[]

def test_low_stock_allows_only_one_guarantee_free_scoped_attempt(monkeypatch):
    monkeypatch.setattr(stock,"query",lambda:packet("Low"))
    assert stock.main()==0
    monkeypatch.setattr(stock,"query",lambda:packet("None"))
    assert stock.main()==75
    monkeypatch.setattr(stock,"query",lambda:packet("High"))
    assert stock.main()==0


def test_validation_failure_is_named_but_never_leaks_provider_message(monkeypatch,capsys):
    monkeypatch.setattr(stock,"query",lambda:{"errors":[{"message":"TOKEN_SUPER_SECRET"}]})
    assert stock.main()==75
    observed=capsys.readouterr().out
    assert "graphql_error_response" in observed
    assert "TOKEN_SUPER_SECRET" not in observed
    assert "confident_capacity_signal" in observed


def test_low_stock_with_null_count_is_eligible_only_for_one_attempt(monkeypatch,capsys):
    data=packet("Low",2.59,(1,))
    data["data"]["gpuTypes"][0]["lowestPrice"]["availableGpuCounts"]=None
    monkeypatch.setattr(stock,"query",lambda:data)
    assert stock.main()==0
    observed=capsys.readouterr().out
    assert '"stock_status": "Low"' in observed
    assert "error_type" not in observed
    assert '"confident_capacity_signal": true' in observed
    assert "limited_capacity_single_attempt_only" in observed


@pytest.mark.parametrize("style",["all_null_quote","null_lowest_price","explicit_none_stock"])
def test_no_regional_capacity_quotes_fail_closed_as_unavailable(style):
    p=packet()
    obj=p["data"]["gpuTypes"][0]
    if style=="null_lowest_price":
        obj["lowestPrice"]=None
    elif style=="all_null_quote":
        obj["lowestPrice"]={
            "stockStatus":None,"uninterruptablePrice":None,
            "availableGpuCounts":None}
    else:
        obj["lowestPrice"]={
            "stockStatus":"None","uninterruptablePrice":None,
            "availableGpuCounts":None}
    result=stock.inspect_response(p)
    assert result["confident_capacity_signal"] is False
    assert result["stock_status"]=="None"
    assert result["availability_reason"] in (
        "no_regional_quote","null_regional_quote","regional_stock_insufficient")


def test_live_all_null_quote_exits_as_capacity_not_schema_error(monkeypatch,capsys):
    p=packet()
    p["data"]["gpuTypes"][0]["lowestPrice"]={
        "stockStatus":None,"uninterruptablePrice":None,
        "availableGpuCounts":None}
    monkeypatch.setattr(stock,"query",lambda:p)
    assert stock.main()==75
    out=capsys.readouterr().out
    assert '"availability_reason": "null_regional_quote"' in out
    assert '"confident_capacity_signal": false' in out
    assert '"stock_status": "None"' in out
    assert "error_code" not in out


def test_partial_missing_quote_fields_fails_closed_without_guessing():
    p=packet()
    del p["data"]["gpuTypes"][0]["lowestPrice"]["stockStatus"]
    with pytest.raises(ValueError,match="missing expected regional quote fields"):
        stock.inspect_response(p)


def test_low_is_not_permission_to_ignore_price_or_missing_region(monkeypatch):
    with pytest.raises(ValueError,match="GPU hourly price unavailable"):
        stock.inspect_response(packet("Low",None,(1,)))
    assert stock.inspect_response(packet("Low",2.60,(1,)))["confident_capacity_signal"] is False
    assert stock.inspect_response(packet("Low",2.59,(2,)))["confident_capacity_signal"] is False
    assert stock.inspect_response(packet("Low",2.59,(1,)))["scarcity_warning"]=="limited_capacity_single_attempt_only"
