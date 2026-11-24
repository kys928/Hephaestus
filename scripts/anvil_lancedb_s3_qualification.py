"""Bounded synthetic qualification of LanceDB over the Czech S3 volume.

No production objects, Pod API, model downloads, wallet, or cleanup operations.
The runner creates only a fresh run/attempt prefix and preserves all evidence.
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import lancedb
import pyarrow as pa

ENDPOINT = "https://s3api-eu-cz-1.runpod.io/"
REGION = "EU-CZ-1"
VOLUME = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
RUN = os.environ.get("GITHUB_RUN_ID", "")
ATTEMPT = os.environ.get("GITHUB_RUN_ATTEMPT", "")
KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY_ID", "")
SECRET = os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY", "")
if VOLUME != "cviwpryzao" or not KEY or not SECRET:
    raise SystemExit("Required scoped Czech volume identity or S3 credentials unavailable")
if not re.fullmatch(r"[0-9]+", RUN) or not re.fullmatch(r"[0-9]+", ATTEMPT):
    raise SystemExit("A GitHub run/attempt identity is required")
PREFIX = "hephaestus/research/anvil-lancedb-s3/" + RUN + "/" + ATTEMPT + "/"
URI = "s3://" + VOLUME + "/" + PREFIX + "db"
OPTIONS = {
    "aws_endpoint": ENDPOINT, "aws_region": REGION,
    "aws_access_key_id": KEY, "aws_secret_access_key": SECRET,
    "aws_virtual_hosted_style_request": "false",
    "timeout": "15s",
}
ROLES = ("controller", "diagnosis", "planner", "coder", "evaluator", "judge")
SCHEMA = pa.schema([
    ("id", pa.string()), ("role", pa.string()), ("investigation_id", pa.string()),
    ("claim", pa.string()), ("sha256", pa.string()), ("vector", pa.list_(pa.float32(), 8)),
])
REPORT = {"schema": "hephaestus.anvil-lancedb-qualification.v1", "run": RUN,
          "attempt": ATTEMPT, "prefix": PREFIX, "synthetic_only": True,
          "tests": {}, "source": os.environ.get("GITHUB_SHA", "")}
RESULT = Path("anvil-lancedb-report.json")
CLIENT = boto3.client("s3", region_name=REGION, endpoint_url=ENDPOINT,
    aws_access_key_id=KEY, aws_secret_access_key=SECRET,
    config=Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 2},
                  s3={"addressing_style": "path"}))

def report(name, status, **fields):
    REPORT["tests"][name] = {"status": status, **fields}
    RESULT.write_text(json.dumps(REPORT, indent=2, sort_keys=True))
    print("RESULT", name, status, json.dumps(fields, sort_keys=True), flush=True)

def safe_error(exc):
    s = str(exc)
    for value in (KEY, SECRET, VOLUME):
        if value:
            s = s.replace(value, "[redacted]")
    # Never echo signed URLs, response bodies or credential-bearing headers.
    return {"error_type": type(exc).__name__, "message": s.splitlines()[0][:240]
            if not any(x in s.lower() for x in ("authorization", "signature", "credential", "http", "secret", "key=")) else "[details suppressed]"}

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def scoped_key(relative):
    if not relative or relative.startswith("/") or ".." in relative.split("/"):
        raise ValueError("Unsafe relative key")
    return PREFIX + relative

def listing(relative=""):
    pages = CLIENT.get_paginator("list_objects_v2").paginate(
        Bucket=VOLUME, Prefix=PREFIX + relative)
    result = {}
    for page in pages:
        for obj in page.get("Contents", []):
            require(obj["Key"].startswith(PREFIX), "Listing escaped experimental prefix")
            result[obj["Key"]] = obj["Size"]
    return result

def put(relative, data, **kw):
    return CLIENT.put_object(Bucket=VOLUME, Key=scoped_key(relative), Body=data, **kw)

def read(relative, **kw):
    obj = CLIENT.get_object(Bucket=VOLUME, Key=scoped_key(relative), **kw)
    with obj["Body"] as body:
        return body.read()

def preflight():
    require(not listing(), "Experimental prefix already contains objects")
    data = b"anvil-synthetic-conditional-write-test"
    put("preflight/basic.bin", data)
    require(read("preflight/basic.bin") == data, "S3 round-trip changed bytes")
    require(read("preflight/basic.bin", Range="bytes=2-6") == data[2:7], "Range read mismatch")
    require(scoped_key("preflight/basic.bin") in listing("preflight/"), "S3 listing missing object")
    report("s3_basic", "pass", bytes=len(data))
    try:
        put("preflight/conditional.bin", b"first", IfNoneMatch="*")
        try:
            put("preflight/conditional.bin", b"second", IfNoneMatch="*")
        except ClientError as e:
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            require(status in (409, 412), "Conditional failure did not report conflict")
        else:
            raise AssertionError("S3 ignored conditional create")
        require(read("preflight/conditional.bin") == b"first", "Conditional create overwrote prior bytes")
        report("s3_conditional_create", "pass")
        return True
    except Exception as e:
        report("s3_conditional_create", "fail", **safe_error(e))
        return False

def row(ident, role, topic=0, claim=None):
    claim = claim if claim is not None else "Synthetic contribution " + ident
    rec = {"id": ident, "role": role, "investigation_id": "synthetic-investigation-001",
           "claim": claim, "vector": [float(i == topic) for i in range(8)]}
    rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rec

BASE = [row("base-" + str(i), role, i) for i, role in enumerate(ROLES)]
def table_data(rows):
    return pa.Table.from_pylist(rows, schema=SCHEMA)

def connect(fresh=True):
    return lancedb.connect(URI, storage_options=OPTIONS,
        read_consistency_interval=datetime.timedelta(seconds=0) if fresh else None)

def table():
    return connect().open_table("contributions")

def verify_rows(actual, expected, allow_extra=False):
    ids = [r["id"] for r in actual]
    require(len(ids) == len(set(ids)), "Duplicate contribution identities")
    by_id = {r["id"]: r for r in actual}
    for r in expected:
        require(by_id.get(r["id"]) == r, "Missing or altered contribution: " + r["id"])
    if not allow_extra:
        require(set(ids) == {r["id"] for r in expected}, "Unexpected contribution identities")
    return by_id

def all_rows(tab):
    return tab.to_arrow().to_pylist()

def worker(mode, ident, ready, gate):
    tab = table()
    Path(ready).touch()
    end = time.monotonic() + 45
    while not Path(gate).exists():
        require(time.monotonic() < end, "Worker gate timeout")
        time.sleep(.03)
    if mode == "concurrent":
        tab.add(table_data([row(ident, "planner", 2)]))
        print("ACK:" + ident, flush=True)
    elif mode == "interrupt":
        tab.add(table_data([row("ack-before-kill", "diagnosis", 1)]))
        print("ACK:ack-before-kill", flush=True)
        # Start a second, larger append. Parent kills after observing new files.
        print("BEGIN:unacknowledged", flush=True)
        batch = [row("pending-" + str(i), "coder", 3,
                     hashlib.sha256(str(i).encode()).hexdigest() * 512)
                 for i in range(128)]
        tab.add(table_data(batch))
        print("ACK:pending-batch", flush=True)
    else:
        raise ValueError("Unknown worker mode")

def wait_ready(paths, procs):
    end = time.monotonic() + 40
    while not all(p.exists() for p in paths):
        require(time.monotonic() < end and all(p.poll() is None for p in procs),
                "Worker did not reach synchronization barrier")
        time.sleep(.05)

def stop(procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
        try:
            p.communicate(timeout=10)
        except Exception:
            pass

def main():
    report("versions", "recorded", lancedb=lancedb.__version__, pyarrow=pa.__version__,
           boto3=boto3.__version__)
    conditional = preflight()
    t = connect().create_table("contributions", data=table_data(BASE), mode="create")
    verify_rows(all_rows(t), BASE)
    report("structured_insert", "pass", rows=len(BASE))
    filtered = t.search().where("role = 'diagnosis' AND investigation_id = 'synthetic-investigation-001'").limit(10).to_list()
    require([r["id"] for r in filtered] == ["base-1"], "Wrong metadata query result")
    near = t.search([0., 0., 1., 0., 0., 0., 0., 0.]).limit(1).to_list()
    require(near[0]["id"] == "base-2", "Wrong nearest neighbor")
    report("metadata_vector_queries", "pass", metadata="base-1", nearest="base-2")
    stale = connect(False).open_table("contributions")
    before = len(all_rows(stale))
    new = row("stale-reader-probe", "judge", 5)
    t.add(table_data([new]))
    immediate = len(all_rows(stale))
    stale.checkout_latest()
    verify_rows(all_rows(stale), BASE + [new])
    report("stale_reader", "pass", old_count=before, immediate_count=immediate,
           refreshed_count=len(all_rows(stale)), explicit_refresh=True)
    expected = BASE + [new]
    if not conditional:
        report("concurrent_submissions", "blocked", reason="Conditional-create semantics failed; no unsafe concurrent writers")
        report("interrupted_write", "blocked", reason="Commit prerequisite failed")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            procs = []
            try:
                paths = [d / ("ready-" + str(i)) for i in range(4)]
                gate = d / "go"
                for i in range(4):
                    procs.append(subprocess.Popen([sys.executable, __file__, "worker",
                        "concurrent", "parallel-" + str(i), str(paths[i]), str(gate)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                wait_ready(paths, procs)
                gate.touch()
                acked = []
                rejected = []
                for i, p in enumerate(procs):
                    out, _ = p.communicate(timeout=90)
                    ident = "parallel-" + str(i)
                    if p.returncode == 0 and "ACK:" + ident in out.splitlines():
                        acked.append(row(ident, "planner", 2))
                    else:
                        rejected.append(ident)
                actual = verify_rows(all_rows(table()), expected + acked, allow_extra=True)
                extras = set(actual) - {r["id"] for r in expected + acked}
                require(extras.issubset(set(rejected)), "Unexpected concurrent rows")
                report("concurrent_submissions", "pass" if len(acked) == 4 else "partial",
                       submitted=4, acknowledged=len(acked), rejected=rejected,
                       unacknowledged_visible=sorted(extras), lost_acknowledged=0)
                for ident in sorted(extras):
                    require(actual[ident] == row(ident, "planner", 2), "Unacknowledged row corrupted")
                    acked.append(actual[ident])
                expected += acked
            finally:
                stop(procs)
        # Separate subprocess: a completed append followed by an unacknowledged append.
        # Kill only after real remote object activity is observed. A late kill is inconclusive.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            log = d / "worker.log"
            with log.open("w") as out:
                p = subprocess.Popen([sys.executable, __file__, "worker", "interrupt",
                    "unused", str(d/"ready"), str(d/"go")],
                    stdout=out, stderr=subprocess.DEVNULL, text=True)
                try:
                    wait_ready([d/"ready"], [p])
                    prior = listing("db/")
                    (d/"go").touch()
                    deadline = time.monotonic() + 75
                    observed = False
                    while time.monotonic() < deadline and p.poll() is None:
                        text = log.read_text()
                        if "BEGIN:unacknowledged" in text:
                            # Exclude files from the first acknowledged append by taking
                            # a new baseline after its acknowledgement.
                            if not observed:
                                prior = listing("db/")
                                observed = True
                            elif set(listing("db/")) - set(prior):
                                break
                        time.sleep(.05)
                    prekill = log.read_text()
                    in_flight = observed and p.poll() is None and "ACK:pending-batch" not in prekill
                    remote_progress = observed and bool(set(listing("db/")) - set(prior))
                    if p.poll() is None:
                        p.kill()
                    p.wait(timeout=10)
                    last = log.read_text()
                    require("ACK:ack-before-kill" in last, "First append not acknowledged")
                    expected.append(row("ack-before-kill", "diagnosis", 1))
                    actual = verify_rows(all_rows(table()), expected, allow_extra=True)
                    pending = {k for k in actual if k.startswith("pending-")}
                    require(len(pending) in (0, 128), "Interrupted batch partially visible")
                    if pending:
                        pending_rows = [row("pending-" + str(i), "coder", 3,
                            hashlib.sha256(str(i).encode()).hexdigest() * 512)
                            for i in range(128)]
                        verify_rows(list(actual.values()), expected + pending_rows)
                        expected += pending_rows
                    else:
                        verify_rows(list(actual.values()), expected)
                    valid = in_flight and remote_progress and "ACK:pending-batch" not in last and p.returncode < 0
                    report("interrupted_write", "pass" if valid else "inconclusive",
                           killed_in_flight=valid, remote_object_progress=remote_progress,
                           acknowledged_first=True, second_batch_visible=len(pending),
                           lost_acknowledged=0)
                finally:
                    stop([p])
    # Store independently generated expected record commitments for the second runner.
    expected_rows = sorted(expected, key=lambda r: r["id"])
    expected_ids = [{"id": r["id"], "sha256": r["sha256"]} for r in expected_rows]
    actual = verify_rows(all_rows(table()), expected_rows)
    report("fresh_connection_reconstruction", "pass", rows=len(actual), lost_acknowledged=0)
    Path("anvil-expected.json").write_text(json.dumps(expected_ids, sort_keys=True, indent=2))
    report("remote_footprint", "recorded", objects=len(listing()), bytes=sum(listing().values()))
    REPORT["status"] = "complete"
    RESULT.write_text(json.dumps(REPORT, indent=2, sort_keys=True))
    if not all(REPORT["tests"].get(k, {}).get("status") == "pass" for k in
        ("s3_basic", "s3_conditional_create", "structured_insert", "metadata_vector_queries",
         "stale_reader", "concurrent_submissions", "interrupted_write", "fresh_connection_reconstruction")):
        raise SystemExit(2)

def independent():
    expected = json.loads(Path("anvil-expected.json").read_text())
    rows = all_rows(table())
    got = []
    for r in rows:
        stored = r.pop("sha256")
        digest = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(digest == stored, "Independent record hash mismatch")
        got.append({"id": r["id"], "sha256": stored})
    require(sorted(got, key=lambda r: r["id"]) == expected, "Independent reconstruction differs from expected commitments")
    report("independent_runner_reconstruction", "pass", rows=len(got), lost_acknowledged=0)
    REPORT["status"] = "complete"
    RESULT.write_text(json.dumps(REPORT, indent=2, sort_keys=True))

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "worker":
            worker(*sys.argv[2:])
        elif len(sys.argv) > 1 and sys.argv[1] == "verify":
            independent()
        else:
            main()
    except Exception as exc:
        if len(sys.argv) <= 1 or sys.argv[1] != "worker":
            report("fatal", "fail", **safe_error(exc))
        sys.exit(1)
