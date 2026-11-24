"""Isolated negative controls after RunPod failed conditional S3 creation.
No production database or artifact is opened. All writes use the run/attempt prefix.
A successful trial cannot override the failed provider commit-safety gate.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import Counter
import anvil_lancedb_s3_qualification as q

BASE = q.BASE
EXPECT = Path("anvil-expected.json")
def open_table(name):
    return q.connect().open_table(name)

def pending_rows():
    return [q.row("pending-" + str(i), "coder", 3,
                  hashlib.shake_256(str(i).encode()).hexdigest(16384))
            for i in range(256)]

def worker(mode, ident, ready, gate, second):
    name = "direct_concurrent" if mode == "parallel" else "interrupted_single"
    t = open_table(name)
    _ = t.count_rows()
    batch = pending_rows() if mode == "interrupt" else []
    Path(ready).touch()
    deadline = time.monotonic() + 60
    while not Path(gate).exists():
        q.require(time.monotonic() < deadline, "Start barrier timeout")
        time.sleep(.02)
    if mode == "parallel":
        t.add(q.table_data([q.row(ident, "planner", 2)]))
        print("ACK:" + ident, flush=True)
    else:
        t.add(q.table_data([q.row("ack-before-kill", "diagnosis", 1)]))
        print("ACK:ack-before-kill", flush=True)
        while not Path(second).exists():
            q.require(time.monotonic() < deadline, "Second barrier timeout")
            time.sleep(.02)
        print("BEGIN:pending", flush=True)
        t.add(q.table_data(batch))
        print("ACK:pending", flush=True)

def digests(rows):
    return {r["id"]: r["sha256"] for r in rows}

def examine(name, expected, possible=()):
    rows = q.all_rows(open_table(name))
    counts = Counter(r["id"] for r in rows)
    actual = {}
    invalid = []
    for r in rows:
        c = dict(r)
        stored = c.pop("sha256")
        computed = hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if computed != stored:
            invalid.append(r["id"])
        actual[r["id"]] = stored
    missing = sorted(set(expected)-set(actual))
    altered = sorted(k for k in expected.keys() & actual.keys() if actual[k] != expected[k])
    unexpected = sorted(set(actual)-set(expected)-set(possible))
    possible_altered = sorted(k for k in possible.keys() & actual.keys() if actual[k] != possible[k]) if isinstance(possible,dict) else []
    dup = sorted(k for k,n in counts.items() if n>1)
    out = {"rows":len(rows), "lost_acknowledged":len(missing), "missing":missing,
           "altered":altered, "duplicate":dup, "invalid_hashes":invalid,
           "unexpected":unexpected, "possible_altered":possible_altered,
           "unacknowledged_visible":sorted(set(actual)-set(expected))}
    return out

def main():
    condition = q.preflight()
    q.report("conditional_write_readback", "pass" if condition else "fail",
             second_write_visible=q.read("preflight/conditional.bin")==b"second")
    q.report("admission", "blocked" if not condition else "not_qualified",
             note="Explicit synthetic-only negative control; not an admission test")
    expected = {}
    possible = {}
    t = q.connect().create_table("direct_concurrent", data=q.table_data(BASE), mode="create")
    expected["direct_concurrent"] = digests(BASE)
    possible["direct_concurrent"] = {}
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        procs=[]
        try:
            paths=[d/("ready-"+str(i)) for i in range(4)]
            gate=d/"go"
            for i in range(4):
                procs.append(subprocess.Popen([sys.executable,__file__,"worker","parallel",
                    "parallel-"+str(i),str(paths[i]),str(gate),str(d/"unused")],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
            q.wait_ready(paths,procs)
            gate.touch()
            acked=[]
            failed=[]
            for i,p in enumerate(procs):
                ident="parallel-"+str(i)
                try:
                    out,err=p.communicate(timeout=90)
                except subprocess.TimeoutExpired:
                    p.kill()
                    out,err=p.communicate(timeout=10)
                row=q.row(ident,"planner",2)
                if "ACK:"+ident in out.splitlines():
                    acked.append(ident)
                    expected["direct_concurrent"].update(digests([row]))
                else:
                    failed.append(ident)
                    possible["direct_concurrent"].update(digests([row]))
            result=examine("direct_concurrent",expected["direct_concurrent"],possible["direct_concurrent"])
            bad=any(result[k] for k in ("missing","altered","duplicate","invalid_hashes","unexpected","possible_altered"))
            q.report("direct_concurrent_negative_control","fail" if bad else ("observed_no_loss_unqualified" if len(acked)==4 else "partial"),
                     submitted=4, acknowledged=len(acked), failed=failed, **result)
        finally:
            q.stop(procs)
    # Deliberately separate table: an unsafe multiwriter trial must not affect
    # single-writer recovery. This is still only a synthetic fault control.
    q.connect().create_table("interrupted_single",data=q.table_data(BASE),mode="create")
    expected["interrupted_single"]=digests(BASE)
    possible["interrupted_single"]=digests(pending_rows())
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        logfile=d/"worker.log"
        with logfile.open("w") as out:
            p=subprocess.Popen([sys.executable,__file__,"worker","interrupt","unused",
                str(d/"ready"),str(d/"go"),str(d/"second")],stdout=out,stderr=subprocess.DEVNULL,text=True)
            try:
                q.wait_ready([d/"ready"],[p])
                (d/"go").touch()
                end=time.monotonic()+45
                while "ACK:ack-before-kill" not in logfile.read_text():
                    q.require(p.poll() is None and time.monotonic()<end,"First append did not acknowledge")
                    time.sleep(.03)
                expected["interrupted_single"].update(digests([q.row("ack-before-kill","diagnosis",1)]))
                old=q.listing("db/interrupted_single.lance/")
                (d/"second").touch()
                deadline=time.monotonic()+60
                progress=False
                while time.monotonic()<deadline and p.poll() is None:
                    if set(q.listing("db/interrupted_single.lance/"))-set(old):
                        progress=True
                        break
                    if "ACK:pending" in logfile.read_text():
                        break
                    time.sleep(.02)
                before=logfile.read_text()
                killed=False
                if p.poll() is None:
                    p.kill()
                    killed=True
                p.wait(timeout=10)
                after=logfile.read_text()
                if "ACK:pending" in after:
                    expected["interrupted_single"].update(possible["interrupted_single"])
                    possible["interrupted_single"]={}
                result=examine("interrupted_single",expected["interrupted_single"],possible["interrupted_single"])
                atomic=len(result["unacknowledged_visible"]) in (0,256)
                valid=killed and progress and "BEGIN:pending" in before and "ACK:pending" not in after and p.returncode<0
                bad=not atomic or any(result[k] for k in ("missing","altered","duplicate","invalid_hashes","unexpected","possible_altered"))
                q.report("interrupted_single_writer","fail" if bad else ("pass" if valid else "inconclusive"),
                         killed_before_ack=valid, remote_object_progress=progress,
                         pending_acknowledged="ACK:pending" in after, atomic_visibility=atomic, **result)
            finally:
                q.stop([p])
    # Serialized writer positive control, separate from direct multi-writer.
    t=q.connect().create_table("serialized",data=q.table_data(BASE),mode="create")
    expected["serialized"]=digests(BASE)
    possible["serialized"]={}
    for i in range(4):
        row=q.row("serialized-"+str(i),"planner",2)
        t.add(q.table_data([row]))
        expected["serialized"].update(digests([row]))
    result=examine("serialized",expected["serialized"],{})
    bad=any(result[k] for k in ("missing","altered","duplicate","invalid_hashes","unexpected"))
    q.report("serialized_writer","fail" if bad else "pass",acknowledged=10,**result)
    EXPECT.write_text(json.dumps({"expected":expected,"possible":possible},indent=2,sort_keys=True))
    q.report("remote_footprint","recorded",objects=len(q.listing()),bytes=sum(q.listing().values()))
    q.REPORT["status"]="complete_negative_control"
    q.RESULT.write_text(json.dumps(q.REPORT,indent=2,sort_keys=True))
    # Qualification is intentionally denied if S3 conditional semantics are broken.
    if not condition or any(v["status"] in ("fail","inconclusive","partial") for v in q.REPORT["tests"].values()):
        sys.exit(2)

def independent():
    e=json.loads(EXPECT.read_text())
    failed=False
    for name,expected in e["expected"].items():
        result=examine(name,expected,e["possible"][name])
        atomic=name!="interrupted_single" or len(result["unacknowledged_visible"]) in (0,256)
        bad=not atomic or any(result[k] for k in ("missing","altered","duplicate","invalid_hashes","unexpected","possible_altered"))
        q.report("independent_"+name,"fail" if bad else "pass",atomic_visibility=atomic,**result)
        failed=failed or bad
    q.REPORT["status"]="complete_negative_control"
    q.RESULT.write_text(json.dumps(q.REPORT,indent=2,sort_keys=True))
    if failed:
        sys.exit(2)

if __name__=="__main__":
    try:
        if len(sys.argv)>1 and sys.argv[1]=="worker":
            worker(*sys.argv[2:])
        elif len(sys.argv)>1 and sys.argv[1]=="verify":
            independent()
        else:
            main()
    except Exception as exc:
        if len(sys.argv)<=1 or sys.argv[1]!="worker":
            q.report("fatal","fail",**q.safe_error(exc))
        sys.exit(1)
