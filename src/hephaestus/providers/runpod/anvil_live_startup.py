"""Bounded, sanitized startup diagnostics. No execution POST or provider mutation."""
from base64 import b64encode
from datetime import datetime, timezone
import errno
import json
import re
import socket
import ssl
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
import zlib

from hephaestus.providers.runpod.anvil_live_transport import NoRedirect


def gateway_launch(source):
    payload = b64encode(zlib.compress(source, 9)).decode()
    code = f"import base64,zlib;exec(zlib.decompress(base64.b64decode('{payload}')))"
    # Explicit override: no dependence on empty-list/default ENTRYPOINT semantics.
    return ["python", "-u", "-c"], [code]


def health_probe(client):
    try:
        result = client.request("/health")
    except HTTPError as exc:
        code = exc.code
        reason = ("http_authentication_or_permission" if code in (401, 403)
                  else "http_proxy_or_upstream_error" if code >= 500
                  else "http_redirect_denied" if 300 <= code < 400
                  else "http_error")
        gateway_denial_shape = False
        if code == 403:
            try:
                gateway_denial_shape = json.loads(exc.read(4097)) == {"status": "denied"}
            except (ValueError, OSError):
                pass
        return {"reason": reason, "http_status": code, "gateway_denial_shape": gateway_denial_shape}
    except json.JSONDecodeError:
        return {"reason": "invalid_json_response", "http_status": 200}
    except (URLError, TimeoutError, OSError) as exc:
        cause = exc.reason if isinstance(exc, URLError) else exc
        reason = ("dns_failure" if isinstance(cause, socket.gaierror)
                  else "tls_failure" if isinstance(cause, ssl.SSLError)
                  else "connection_refused" if isinstance(cause, ConnectionRefusedError)
                  else "connection_reset" if isinstance(cause, ConnectionResetError)
                  else "timeout" if isinstance(cause, TimeoutError)
                  else "network_unreachable" if getattr(cause, "errno", None) in (errno.ENETUNREACH, errno.EHOSTUNREACH)
                  else "transport_error")
        return {"reason": reason, "http_status": None}
    except (ValueError, UnicodeError):
        return {"reason": "invalid_or_oversized_response", "http_status": None}
    return {"reason": "ready" if result == {"status": "ready"} else "unexpected_health_schema",
            "http_status": 200}


def provider_state(pod):
    if not isinstance(pod, dict):
        return {"status": "unobserved"}
    allowed = {"RUNNING", "INITIALIZING", "PROVISIONING", "STARTING", "STOPPED",
               "EXITED", "TERMINATED", "ERROR", "CREATED", "PENDING", "UNKNOWN"}
    row = {}
    for key in ("status", "desiredStatus", "runtimeStatus"):
        val = pod.get(key)
        row[key] = val.upper() if isinstance(val, str) and val.upper() in allowed else "UNKNOWN"
    row["runtime_reported"] = isinstance(pod.get("runtime"), dict)
    ports = pod.get("ports")
    row["http_port_declared"] = "8787/http" in ports if isinstance(ports, list) else None
    return row


def startup_log_signals(pod_id, token, deadline):
    """Read exact provider SSE streams before teardown; never retain raw lines."""
    if not re.fullmatch(r"[A-Za-z0-9]{4,40}", pod_id):
        raise ValueError("exact Pod identity required")
    opener = build_opener(NoRedirect())
    reports = []
    for source in ("system", "container"):
        row = {"source": source, "http_status": None, "events": 0, "signals": []}
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 1:
            row["error_type"] = "Deadline"
            reports.append(row)
            continue
        limit = monotonic() + min(8, remaining)
        request = Request(f"https://api.runpod.io/v2/pods/{pod_id}/logs?source={source}&tail=40",
                          headers={"Authorization": "Bearer " + token, "Accept": "text/event-stream"})
        try:
            with opener.open(request, timeout=min(5, remaining)) as response:
                row["http_status"] = response.status
                while monotonic() < limit and row["events"] < 40:
                    raw = response.readline(65537)
                    if not raw:
                        break
                    if len(raw) > 65536 or not raw.startswith(b"data:"):
                        continue
                    event = json.loads(raw[5:])
                    line = str(event.get("line", "")).lower()
                    row["events"] += 1
                    signals = {
                        "bootstrap_entered": ("anvil_gateway_event=bootstrap_entered",),
                        "env_validated": ("anvil_gateway_event=env_validated",),
                        "listener_bound": ("anvil_gateway_event=listener_bound",),
                        "bootstrap_failed": ("anvil_gateway_event=bootstrap_failed:",),
                        "image_pull": ("pulling image", "pull image", "image pull"),
                        "container_start": ("start container", "starting container"),
                        "entrypoint_error": ("executable file not found", "exec format error", "permission denied"),
                        "python_traceback": ("traceback (most recent call last)", "modulenotfounderror", "keyerror:"),
                        "oom": ("out of memory", "oomkilled"),
                        "port_bind_error": ("address already in use", "cannot assign requested address"),
                    }
                    for label, needles in signals.items():
                        if any(n in line for n in needles) and label not in row["signals"]:
                            row["signals"].append(label)
        except HTTPError as exc:
            row["http_status"] = exc.code
        except (URLError, OSError, ValueError, AttributeError) as exc:
            row["error_type"] = type(exc).__name__
        reports.append(row)
    return reports


class WorkerHealthFailure(TimeoutError):
    def __init__(self, observations, attempts, reason="worker_health_deadline"):
        super().__init__("exact worker never became healthy")
        self.diagnostic = {"stage": "worker_health", "reason": reason,
                           "exception": type(self).__name__, "attempts": attempts,
                           "observations": observations, "http_status": (
                               observations[-1]["health"]["http_status"] if observations else None)}
