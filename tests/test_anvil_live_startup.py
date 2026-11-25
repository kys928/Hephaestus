"""Startup failure classification and launch regression (no GPU/provider calls)."""
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import socket
import ssl
from urllib.error import HTTPError, URLError
import pytest

from hephaestus.providers.runpod.anvil_live_startup import gateway_launch, health_probe, provider_state


class Client:
    def __init__(self, result):
        self.result = result
    def request(self, path):
        assert path == "/health"
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("error,reason", [
    (URLError(socket.gaierror(-2, "private")), "dns_failure"),
    (URLError(ssl.SSLError("private")), "tls_failure"),
    (URLError(ConnectionRefusedError()), "connection_refused"),
    (URLError(ConnectionResetError()), "connection_reset"),
    (TimeoutError(), "timeout"),
    (HTTPError("https://private", 503, "private", {}, BytesIO(b"private")), "http_proxy_or_upstream_error"),
    (HTTPError("https://private", 403, "private", {}, BytesIO(b"private")), "http_authentication_or_permission"),
    (HTTPError("https://private", 302, "private", {}, BytesIO()), "http_redirect_denied"),
    (json.JSONDecodeError("private", "private", 0), "invalid_json_response"),
])
def test_health_failure_distinguished_and_sanitized(error, reason):
    row = health_probe(Client(error))
    assert row["reason"] == reason
    assert "private" not in json.dumps(row)


def test_health_contract_and_auth_shape():
    assert health_probe(Client({"status":"ready"}))["reason"] == "ready"
    assert health_probe(Client({"status":"other"}))["reason"] == "unexpected_health_schema"
    denied = HTTPError("https://private", 403, "private", {}, BytesIO(b'{"status":"denied"}'))
    assert health_probe(Client(denied))["gateway_denial_shape"] is True


def test_explicit_launch_executes_exact_bytes():
    import subprocess
    entrypoint,cmd=gateway_launch(b'print("exact-bootstrap")')
    assert entrypoint == ["python","-u","-c"] and len(cmd)==1
    assert subprocess.check_output(entrypoint+cmd,text=True).strip()=="exact-bootstrap"


def test_provider_state_does_not_equate_desired_and_actual():
    row=provider_state({"desiredStatus":"RUNNING","runtimeStatus":"starting","env":{"secret":"private"},"ports":["8787/http"]})
    assert row["desiredStatus"]=="RUNNING" and row["runtimeStatus"]=="STARTING"
    assert row["runtime_reported"] is False and row["http_port_declared"] is True
    assert "private" not in json.dumps(row)
