"""The authorized read must retain partial progress and issue no retries."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/anvil_verify_planner_shards.py"
_spec = importlib.util.spec_from_file_location("anvil_verify_planner_shards", _SCRIPT)
_transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transport)


def test_planner_partial_transport_counts_delivered_bytes_without_retry():
    class Interrupted(io.BytesIO):
        def read1(self, size):
            if self.tell():
                raise ConnectionError("fixture stream interrupted")
            return super().read1(1)

    class S3:
        def get_object(self, *, Range):
            ranges.append(Range)
            first, last = map(int, Range.removeprefix("bytes=").split("-"))
            stream = io.BytesIO(payload[first:last + 1]) if first == 0 else Interrupted(payload[first:last + 1])
            return {
                "Body": stream, "ContentLength": last - first + 1,
                "ContentRange": f"bytes {first}-{last}/{len(payload)}",
                "ResponseMetadata": {"HTTPStatusCode": 206},
            }

    payload = b"range-data"
    digest = hashlib.sha256(payload).hexdigest()
    ranges, events = [], []
    with pytest.raises(ConnectionError, match="interrupted"):
        _transport.verify_s3_ranges(S3(), {}, len(payload), digest, 4, progress=lambda **row: events.append(row))
    assert ranges == ["bytes=0-3", "bytes=4-7"]
    assert [row["bytes"] for row in events if row["event"] == "range_read"] == [4, 1]
