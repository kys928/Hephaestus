"""Transport fixtures, not CUDA or model behavior evidence."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import tarfile
from threading import Event
from time import sleep

import pytest

from hephaestus.providers.runpod.anvil_live_transport import AnvilGatewayClient
from urllib.error import URLError

SPEC = spec_from_file_location("gateway", Path(__file__).resolve().parents[1] / "scripts/anvil_live_pod_gateway.py")
gateway = module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


def archive(name):
    blob = BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as t:
        member = tarfile.TarInfo(name)
        member.size = 1
        t.addfile(member, BytesIO(b"x"))
    return blob.getvalue()


def test_archive_digest_and_path_fence(tmp_path):
    raw = archive("src/example.py")
    gateway.extract_source(raw, tmp_path / "ok", sha256(raw).hexdigest())
    assert (tmp_path / "ok/src/example.py").read_bytes() == b"x"
    with pytest.raises(ValueError, match="pinned"):
        gateway.extract_source(raw, tmp_path / "bad", "0" * 64)
    bad = archive("../escape")
    with pytest.raises(ValueError, match="unsafe"):
        gateway.extract_source(bad, tmp_path / "escape", sha256(bad).hexdigest())


def test_transport_consumes_once_and_requires_predecessor(tmp_path):
    release = Event()
    calls = []
    def executor(name, raw):
        calls.append(name)
        if name == "setup":
            assert release.wait(5)
        return {"fixture": True}
    deadline = datetime.now(timezone.utc) + timedelta(minutes=1)
    g = gateway.Gateway(tmp_path / "transport", "a" * 64, deadline, executor)
    assert g.authorized("Bearer " + "a" * 64) and not g.authorized("wrong")
    with pytest.raises(PermissionError):
        g.start("qualify", b"{}")
    assert g.start("setup", b"test")
    assert not g.start("setup", b"test")
    release.set()
    for _ in range(100):
        if g.status("setup")["status"] == "complete":
            break
        sleep(.01)
    assert g.status("setup")["status"] == "complete"
    reopened = gateway.Gateway(g.root, g.token, deadline, executor)
    assert not reopened.start("setup", b"test")
    assert calls == ["setup"]


def test_lost_post_acknowledgement_does_not_retry():
    client = AnvilGatewayClient("owned1234", "a" * 64, datetime.now(timezone.utc) + timedelta(seconds=5))
    calls = []
    def request(path, raw=None):
        calls.append((path, raw))
        if raw is not None:
            raise URLError("lost ack")
        return {"status": "complete", "result": {"fixture": True}}
    client.request = request
    assert client.execute("slot-0", b"fixture") == {"fixture": True}
    assert [p for p, raw in calls if raw is not None] == ["/execute/slot-0"]
