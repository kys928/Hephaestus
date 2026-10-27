#!/usr/bin/env python3
"""Bounded metadata-only S3 activation check; NEVER authorizes a scientific action.

HEAD pinned frozen role adapter objects and tokenizer, and list a capped
scientific prefix to locate POSSIBLE persisted campaign state. Never GET files,
read source datasets, inspect private holdouts, launch compute or write S3.
An object existing is not independent scientific verification of its bytes.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path

import boto3
from botocore.config import Config

from hephaestus.providers.models.role_stack import (
    load_certified_role_model_stack, REQUIRED_ROLES,
)

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/"configs/models/hephaestus_role_model_stack_v1.json"
HISTORICAL_SELECTION=ROOT/"docs/evidence/first-selected-dataset-preparation-001-33881924136/dataset_approval.json"
TOKENIZER_OBJECT=(
    "hephaestus/scientific/v1/runs/"
    "first-bounded-scientific-training-001-33866198758/"
    "checkpoint_step_100/tokenizer/tokenizer.json"
)
EXPECTED_TOKENIZER_BYTES=0  # HEAD may prove presence, never byte identity


def main()->None:
    stack=load_certified_role_model_stack(REGISTRY)
    bucket=os.environ["RUNPOD_NETWORK_VOLUME_ID"]
    client=boto3.client("s3",
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4",retries={"max_attempts":3,"mode":"standard"}))

    def head(key:str,expected:int|None=None)->dict:
        from botocore.exceptions import ClientError
        try:
            result=client.head_object(Bucket=bucket,Key=key)
        except ClientError as exc:
            code=str(exc.response.get("Error",{}).get("Code","unknown"))
            return {"accessible":False,"error_code":code}
        length=int(result["ContentLength"])
        return {"accessible":True,"observed_bytes":length,
                "byte_count_matches_registry":length==expected if expected is not None else None,
                "cryptographic_content_identity_verified":False}

    objects={role:head(stack.resolve(role).adapter.s3_key,
                        stack.resolve(role).adapter.bytes)
             for role in REQUIRED_ROLES}
    objects["frozen_original_training_tokenizer"]=head(TOKENIZER_OBJECT)

    checked=0
    marker_candidates=[]
    truncated=False
    list_status="unknown"
    # Capped listing is a locator-only study, not permission to read private
    # campaign state and NOT proof the state exists outside scanned prefixes.
    try:
        for prefix in ("hephaestus/scientific/v1/","hephaestus/staged/"):
            token=None
            for page in range(3):
                args={"Bucket":bucket,"Prefix":prefix,"MaxKeys":1000}
                if token:
                    args["ContinuationToken"]=token
                result=client.list_objects_v2(**args)
                rows=result.get("Contents",[])
                checked+=len(rows)
                for item in rows:
                    key=str(item["Key"])
                    lower=key.lower()
                    if (lower.endswith((".sqlite",".sqlite3",".db",".sqlite.gz"))
                            or "staged_workflow_states" in lower
                            or "governed_campaign_state" in lower):
                        marker_candidates.append({
                            "key_sha256":"sha256:"+sha256(key.encode()).hexdigest(),
                            "size_bytes":int(item["Size"]),
                            "possible_staged_state_only":True,
                        })
                token=result.get("NextContinuationToken")
                if not token:
                    break
                if page==2:
                    truncated=True
        list_status="bounded_metadata_listing_completed"
    except Exception as exc:
        # The exception name alone avoids leaking keys or authorization headers.
        list_status="metadata_listing_failed:"+type(exc).__name__

    historical=json.loads(HISTORICAL_SELECTION.read_text())
    grant_types=list(historical.get("approved_requirements",[]))
    report={
        "schema":"hephaestus-real-storage-activation-metadata-only.v1",
        "registry":"configs/models/hephaestus_role_model_stack_v1.json",
        "registered_roles":list(REQUIRED_ROLES),
        "registered_role_adapter_head_results":objects,
        "head_checks_only_not_full_bytes":True,
        "historical_operator_approval_requirements":grant_types,
        "historical_operator_acquisition_approved":"dataset_acquisition" in grant_types,
        "historical_selected_dataset_is_current_scoped_training_grant":False,
        "scientific_prefix_metadata_listing":{
            "status":list_status,"objects_listed":checked,
            "max_objects":6000,"partial_listing":truncated,
            "possible_campaign_state_objects":marker_candidates[:20],
            "found_verified_campaign_state":False,
        },
        "no_full_corpus_acquisition":True,
        "no_model_inference_or_gpu":True,
        "no_dataset_training_authority_created":True,
        "no_campaign_stage_execution":True,
        "not_a_completed_45_stage_trace":True,
    }
    out=Path("governed_activation_storage_metadata.json")
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "roles_registry_head_found":sum(objects[r]["accessible"] for r in REQUIRED_ROLES),
        "roles_registry_size_matching":sum(objects[r]["byte_count_matches_registry"] for r in REQUIRED_ROLES),
        "tokenizer_head_present":objects["frozen_original_training_tokenizer"]["accessible"],
        "historical_acquisition_approval":False,
        "candidate_campaign_state_count":len(marker_candidates),
        "metadata_listing_status":list_status,
        "not_sufficient_to_resume":True,
    },sort_keys=True))


if __name__=="__main__":
    main()
