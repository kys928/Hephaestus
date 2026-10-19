"""Test safe schema-only provider diagnosis, no secrets or networking."""
import diagnose_D_contrast_stock_readonly as d
import preflight_D_contrast_region_stock as gate
from test_preflight_D_contrast_region_stock import packet

def test_null_lowestprice_is_explicit():
    p=packet();p["data"]["gpuTypes"][0]["lowestPrice"]=None
    out=d.shape(p)
    assert out["shape_error"]=="lowestPrice_null"
    assert out["capacity_usable"] is False

def test_no_sensitive_fields_disclosed():
    p=packet()
    p["accountId"]="secret-account-id"
    p["data"]["gpuTypes"][0]["secret"]="private"
    p["data"]["gpuTypes"][0]["lowestPrice"]["internalToken"]="private"
    result=d.shape(p)
    raw=str(result)
    assert "secret-account-id" not in raw and "private" not in raw

def test_legacy_valid_stock_shape_is_recorded():
    out=d.shape(packet("Low",2.49,(1,)))
    assert out["stock_status"]=="Low"
    assert out["gpu_counts_include_one"]
    assert out["price_at_most_2_49"]

def test_graphql_errors_summarized_without_sensitive_messages():
    out=d.shape({"errors":[{"message":"SECRET provider error"}]})
    assert out["shape_error"]=="graphql_errors"
    assert "SECRET" not in str(out)
