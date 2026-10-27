#!/usr/bin/env python3
"""Recover exactly the original 2026-10-25 11-step CI *evidence*, not its state.

Source run 40216554433 archived two JSON reports but NOT the original
temporary pytest SQLite DB and content-addressed receipts. This verifier
refuses to turn a 45-slot status vector into a resumable campaign.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import os
from urllib.request import Request, urlopen
import zipfile

RUN_ID = 40216554433
ARTIFACT_ID = 12607209791
ZIP_SHA256 = "db2963b6bbceb607260448f3a5eadf7e9f02e61b6e8267b499ca0e98cbaf61f4"
MEMBERS = {
    "cpu-real-pretrained-causal-role-output.json":
    "b139b0c91d86e89998db461f42466eb6989b75828c24a74e2ae66e03006ecc0b",
    "cpu-staged-causal-planner.json":
    "2eef80e12c31424e8f556a0a51532ed4c1b45d32134083cbfb443d1d78af1eb1",
}


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def recover(data:bytes)->dict:
    if sha(data)!=ZIP_SHA256:
        raise ValueError("original artifact ZIP byte identity differs")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if set(archive.namelist()) != set(MEMBERS):
            raise ValueError("historical artifact did not preserve exactly its published members")
        payload={}
        for name, expected in MEMBERS.items():
            info=archive.getinfo(name)
            if info.file_size>16_000 or info.is_dir():
                raise ValueError("historical evidence member exceeds bounded size")
            content=archive.read(name)
            if sha(content)!=expected:
                raise ValueError("historical evidence member hash drift")
            payload[name]=json.loads(content)
    stages=payload["cpu-staged-causal-planner.json"]
    model=payload["cpu-real-pretrained-causal-role-output.json"]
    values=stages.get("operation_statuses",{})
    if (len(values)!=45 or sum(x=="completed" for x in values.values())!=11
            or values.get("data_acquisition_audit.acquisition_approval_gate")!="approval_pending"
            or stages.get("full_45_scientific_campaign") is not False
            or model.get("actual_pretrained_causal_lm_inference") is not True
            or model.get("full_45_operation_campaign") is not False):
        raise ValueError("original 11/45 evidence does not match archival scope")
    return {
        "schema":"hephaestus-historical-eleven-stage-status-recovery.v1",
        "source_github_actions_run":RUN_ID,
        "original_artifact_id":ARTIFACT_ID,
        "original_zip_sha256":"sha256:"+ZIP_SHA256,
        "original_member_sha256":{k:"sha256:"+v for k,v in MEMBERS.items()},
        "original_measured_completed_operation_count":11,
        "original_next_operation":"data_acquisition_audit.acquisition_approval_gate",
        "original_pending_status":"approval_pending",
        "actual_original_source_kind":"temporary_pytest_cpu_smoke",
        "original_sqlite_state_in_archive":False,
        "original_content_addressed_operation_store_in_archive":False,
        "original_full_45_campaign":False,
        "historical_control_receipt_chain_reconstructed":False,
        "original_workflow_resumable_from_this_archive":False,
        "new_campaign_started":False,
        "not_an_acquisition_authorization":True,
    }


def main()->None:
    token=os.environ.get("GITHUB_TOKEN","")
    if not token:
        raise PermissionError("read-only GitHub Actions token needed to fetch immutable run evidence")
    request=Request(
        f"https://api.github.com/repos/kys928/Hephaestus/actions/artifacts/{ARTIFACT_ID}/zip",
        headers={"Authorization":"Bearer "+token,
                 "Accept":"application/vnd.github+json",
                 "User-Agent":"hephaestus-original-state-recovery/1"},
    )
    with urlopen(request,timeout=30) as response:
        blob=response.read(50_000+1)
    if len(blob)>50_000:
        raise ValueError("historical artifact exceeds byte ceiling")
    report=recover(blob)
    p=Path("historical_campaign_recovery.json")
    p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
