"""Bounded streaming transport boundary used by remote dataset acquisition."""

from __future__ import annotations

import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AcquisitionTransportError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class StreamingDownloadResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...
    def close(self) -> None: ...


class DownloadTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> StreamingDownloadResponse: ...


@dataclass(slots=True)
class UrllibDownloadTransport:
    user_agent: str = "hephaestus-production-data-acquisition/1"

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> StreamingDownloadResponse:
        request_headers = {
            "Accept": "application/octet-stream",
            "User-Agent": self.user_agent,
            **headers,
        }
        try:
            return urlopen(
                Request(url, headers=request_headers), timeout=timeout_seconds
            )  # type: ignore[return-value]
        except HTTPError as exc:
            code, retryable = _http_failure(exc.code)
            raise AcquisitionTransportError(
                code,
                f"remote transfer failed with HTTP {exc.code}",
                retryable=retryable,
            ) from exc
        except TimeoutError as exc:
            raise AcquisitionTransportError(
                "transfer_timeout", "remote transfer timed out", retryable=True
            ) from exc
        except URLError as exc:
            reason = (
                "timeout"
                if isinstance(exc.reason, socket.timeout)
                else type(exc.reason).__name__
            )
            code = "transfer_timeout" if reason == "timeout" else "provider_unavailable"
            raise AcquisitionTransportError(
                code, f"remote transfer unavailable: {reason}", retryable=True
            ) from exc
        except OSError as exc:
            raise AcquisitionTransportError(
                "connection_interrupted",
                "remote transfer connection failed",
                retryable=True,
            ) from exc


def _http_failure(status: int) -> tuple[str, bool]:
    if status == 401:
        return "authentication_failure", False
    if status == 403:
        return "gated_access_denied", False
    if status == 404:
        return "remote_file_not_found", False
    if status == 408:
        return "transfer_timeout", True
    if status == 429:
        return "rate_limited", True
    if status >= 500:
        return "provider_unavailable", True
    return "provider_transfer_error", False


__all__ = [
    "AcquisitionTransportError",
    "DownloadTransport",
    "StreamingDownloadResponse",
    "UrllibDownloadTransport",
]
