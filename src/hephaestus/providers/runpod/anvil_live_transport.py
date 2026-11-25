"""Exact HTTPS worker transport; never retry an execution POST."""
from datetime import datetime, timezone
import json
import re
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class AnvilGatewayClient:
    def __init__(self, pod_id, token, deadline):
        if (not re.fullmatch(r"[A-Za-z0-9]{4,40}", pod_id)
                or not re.fullmatch(r"[a-f0-9]{64}", token) or deadline.tzinfo is None):
            raise ValueError("exact private worker transport identity required")
        self.base = f"https://{pod_id}-8787.proxy.runpod.net"
        self.token, self.deadline = token, deadline
        self.opener = build_opener(NoRedirect())

    def request(self, path, raw=None):
        remaining = (self.deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("worker transport deadline expired")
        req = Request(self.base + path, data=raw, headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/octet-stream",
        }, method="GET" if raw is None else "POST")
        with self.opener.open(req, timeout=min(20, remaining)) as response:
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("worker result exceeds payload ceiling")
            return json.loads(body)

    def execute(self, name, raw):
        if name not in {"setup", "qualify", "slot-0", "slot-1", "slot-2", "slot-3"}:
            raise ValueError("unknown bounded worker operation")
        try:
            self.request("/execute/" + name, raw)
        except HTTPError as exc:
            if exc.code != 409 and exc.code < 500:
                raise
        except (URLError, TimeoutError, OSError):
            # Ambiguous POST: observe, never POST again.
            pass
        while datetime.now(timezone.utc) < self.deadline:
            try:
                result = self.request("/result/" + name)
            except (URLError, TimeoutError, OSError):
                sleep(3)
                continue
            if result.get("status") == "complete":
                return result["result"]
            if result.get("status") == "failed":
                raise RuntimeError("worker operation failed; no automatic retry")
            if result.get("status") not in {"running", "not_started"}:
                raise ValueError("unrecognized bounded worker response")
            sleep(3)
        raise TimeoutError("worker result unresolved at deadline; no automatic retry")
