from __future__ import annotations

import hashlib
import importlib.util
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ENTRYPOINT = SCRIPTS / "run_interface_repair_v2_cached.py"


def import_entrypoint():
    scripts = str(SCRIPTS)
    inserted = scripts not in sys.path
    if inserted:
        sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("interface_repair_v2_cached_test", ENTRYPOINT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(scripts)


class _Body:
    def __init__(self, raw: bytes) -> None:
        self._stream = BytesIO(raw)

    def read(self) -> bytes:
        return self._stream.read()

    def close(self) -> None:
        self._stream.close()


class FakeClient:
    def __init__(self, *, preserve_metadata: bool) -> None:
        self.preserve_metadata = preserve_metadata
        self.raw = b""
        self.metadata: dict[str, str] = {}
        self.get_calls = 0

    def put_object(self, *, Bucket, Key, Body, ContentType, Metadata):  # noqa: N803
        del Bucket, Key, ContentType
        self.raw = bytes(Body)
        self.metadata = dict(Metadata)

    def head_object(self, *, Bucket, Key):  # noqa: N803
        del Bucket, Key
        return {
            "ContentLength": len(self.raw),
            "Metadata": self.metadata if self.preserve_metadata else {},
        }

    def get_object(self, *, Bucket, Key):  # noqa: N803
        del Bucket, Key
        self.get_calls += 1
        return {"Body": _Body(self.raw)}


def test_v2_s3_verifier_uses_head_metadata_when_available(monkeypatch) -> None:
    module = import_entrypoint()
    monkeypatch.setattr(module.v2.v1, "bucket", lambda: "bucket")
    client = FakeClient(preserve_metadata=True)
    raw = b"abc123"
    result = module.put_bytes_verified_compatible(client, "x.json", raw, "application/json")
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["verification"] == "head_metadata"
    assert client.get_calls == 0


def test_v2_s3_verifier_falls_back_to_get_sha_when_metadata_is_dropped(monkeypatch) -> None:
    module = import_entrypoint()
    monkeypatch.setattr(module.v2.v1, "bucket", lambda: "bucket")
    client = FakeClient(preserve_metadata=False)
    raw = b"payload"
    result = module.put_bytes_verified_compatible(client, "x.json", raw, "application/json")
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["verification"] == "get_sha256_fallback"
    assert client.get_calls == 1
