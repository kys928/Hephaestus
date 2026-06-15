from __future__ import annotations

from hephaestus.app.console import make_handler
from hephaestus.cli.create_demo_state import create_demo_state
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def test_operator_console_is_read_only(tmp_path):
    root = tmp_path / "state"
    create_demo_state(root, "demo-run")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/api/runs") as response:
            body = response.read().decode("utf-8")
        assert "demo-run" in body
        try:
            urlopen(Request(f"{base}/api/runs", data=b"{}", method="POST"))
        except HTTPError as exc:
            assert exc.code == 405
            assert "read_only" in exc.read().decode("utf-8")
        else:  # pragma: no cover
            raise AssertionError("POST unexpectedly succeeded")
    finally:
        server.shutdown()
        thread.join(timeout=5)
