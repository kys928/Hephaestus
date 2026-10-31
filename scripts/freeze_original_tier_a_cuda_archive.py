#!/usr/bin/env python3
"""Freeze one completed native CUDA 45 workspace into byte-verifiable evidence.

No scientific claims are introduced here: a genuine native report with
verified 45/45 and actual CUDA receipts must ALREADY exist. The archive is
private, immutable and intended for independent cold readback, not Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from zipfile import ZipFile,ZIP_STORED

def hash_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(workspace, destination):
    root=Path(workspace).resolve()
    dest=Path(destination).resolve()
    if not root.is_dir() or dest.exists() or dest==root or root in dest.parents:
        raise PermissionError("fresh archive destination must be outside original workspace")
    reports=list(root.glob("cuda-native-report-*.json"))
    completed=[]
    for path in reports:
        if hash_file(path)!=path.stem.removeprefix("cuda-native-report-"):
            raise PermissionError("original SHA named GPU report was modified")
        report=json.loads(path.read_bytes())
        if (report.get("status")=="completed"
                and report.get("completed_operations")==45
                and isinstance(report.get("verified"),dict)
                and report["verified"].get("native_operations")==45
                and report["verified"].get("actual_cuda_optimizer_steps")==96
                and len(report["verified"].get("original_five_cuda_role_inferences",[]))==5
                and report["verified"].get("independent_certification") is False):
            completed.append(report)
    if len(completed)!=1:
        raise PermissionError("exactly one previously verified authentic CUDA 45/45 report required")
    db=root/"state.sqlite"
    if not db.is_file():
        raise PermissionError("original GPU scientific SQLite missing")
    connection=sqlite3.connect(db)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if connection.execute("PRAGMA quick_check").fetchone()[0]!="ok":
            raise PermissionError("original SQLite quick_check failed")
    finally:
        connection.close()
    index={}
    for p in sorted(root.rglob("*")):
        rel=p.relative_to(root).as_posix()
        st=p.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError("no symlinks allowed in original GPU science custody: "+rel)
        if p.is_dir():
            continue
        if (not stat.S_ISREG(st.st_mode)
                or rel in {"archive_sha256.json","state.sqlite-wal","state.sqlite-shm"}
                or rel.endswith(".partial")):
            raise PermissionError("unexpected transient/nonregular original GPU artifact: "+rel)
        index[rel]=hash_file(p)
    if not index or "state.sqlite" not in index:
        raise PermissionError("original GPU scientific archive inventory empty")
    manifest=json.dumps(index,sort_keys=True,indent=2)+"\n"
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+".partial")
    if tmp.exists():
        raise FileExistsError("prior unfinished archive requires reconciliation")
    try:
        with ZipFile(tmp,"x",compression=ZIP_STORED,allowZip64=True) as out:
            out.writestr("archive_sha256.json",manifest)
            for rel in index:
                out.write(root/rel,arcname=rel)
        os.replace(tmp,dest)
    except BaseException:
        # The .partial archive remains for human inspection. Do not overwrite.
        raise
    return {"status":"frozen_original_gpu_45_archive",
            "sha256":"sha256:"+hash_file(dest),
            "member_count":len(index),"bytes":dest.stat().st_size,
            "independent_certification":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.workspace,a.archive),sort_keys=True))

if __name__=="__main__":
    main()
