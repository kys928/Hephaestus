#!/usr/bin/env python3
"""Alternate one-source continuation preview if programmatic math failed.

Public pinned HF *viewer* first rows only. Original tokenizer, no threshold
change, no full corpus acquisition, source approval or holdout clearance.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.parse import quote,urlencode

from tokenizers import Tokenizer
from screen_single_hf_math_source_public_preview import (
    get_json, TOKENIZER_KEY, OLD_BYTES_SHA, OLD_TOKENIZER,
)

DATASET="roneneldan/TinyStories"
EXPECTED_LICENSE="cdla-sharing-1.0"
PREFIX=128
TARGET=32
STRIDE=64


def screen(meta:dict, splits:dict, preview:dict,tokenizer:Tokenizer)->dict:
    rev=str(meta.get("sha",""))
    license=str(meta.get("cardData",{}).get("license","") or meta.get("license",""))
    if not re.fullmatch(r"[0-9a-f]{40}",rev) or license!=EXPECTED_LICENSE:
        raise ValueError("pinned TinyStories revision or declared CDLA license changed")
    identifiers={(s.get("config"),s.get("split")) for s in splits.get("splits",[])
                 if s.get("dataset")==DATASET}
    if ("default","train") not in identifiers:
        raise ValueError("TinyStories default training split missing")
    if (preview.get("dataset"),preview.get("config"),preview.get("split"))!=(DATASET,"default","train"):
        raise ValueError("source preview response identity mismatch")
    unk_id=tokenizer.token_to_id("<|unk|>")
    if unk_id is None:
        raise ValueError("original frozen tokenizer lacks expected unknown token")
    counts=Counter()
    total=bad=0
    pair_count=0
    dedup=set()
    for entry in preview.get("rows",[])[:100]:
        row=entry.get("row",{})
        if entry.get("truncated_cells") or not isinstance(row,dict):
            counts["truncated_or_invalid"]+=1
            continue
        raw=row.get("text")
        if not isinstance(raw,str) or not raw.strip() or len(raw.encode("utf-8"))>20_000:
            counts["empty_or_oversized"]+=1
            continue
        ids=tokenizer.encode(raw).ids
        total+=len(ids)
        bad+=ids.count(unk_id)
        key=sha256(" ".join(raw.casefold().split()).encode()).hexdigest()
        if key in dedup:counts["exact_normalized_repeat"]+=1
        dedup.add(key)
        if re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}",raw,re.I):counts["potential_email"]+=1
        if len(ids)>=PREFIX+TARGET:
            # Count windows where prompt and target lie inside one source row.
            # No EOS, no interpolation over adjacent stories.
            windows=1+(len(ids)-(PREFIX+TARGET))//STRIDE
            pair_count+=windows
            counts["long_enough_for_continuation"]+=1
        counts["valid_rows"]+=1
    frac=bad/max(total,1)
    return {
        "schema":"hephaestus-tinystories-public-continuation-screen.v1",
        "dataset_id":DATASET,"revision":rev,
        "source_license_metadata":license,
        "rights": "publisher-declared-CDLA-Sharing-1.0; obligations and source chain not independently cleared",
        "generated_origin":"publisher-described GPT-3.5 and GPT-4 stories, not verified through samples",
        "preview_scanned_rows":counts["valid_rows"],
        "counts":dict(counts),
        "unique_normalized_preview_rows":len(dedup),
        "original_tokenizer_ref":OLD_TOKENIZER,
        "raw_source_tokenizer":{"total_tokens":total,"unknown_tokens":bad,
                                "unknown_fraction":frac,
                                "passes_original_one_percent_limit":total>0 and frac<=.01},
        "continuation_window":{"prefix_tokens":PREFIX,"target_tokens":TARGET,
                               "stride":STRIDE,"complete_in_source_windows":pair_count,
                               "new_eos_inserted":False},
        "fit_for_original_diagnosis":"next_token_continuation_only_not_structured_json_or_bounded_short_answers",
        "full_corpus_privacy_and_duplicates_cleared":False,
        "independent_protected_evaluation_overlap_checked":False,
        "license_attribution_and_share_alike_approved":False,
        "complete_training_corpus_scanned":False,
        "selection_approved":False,"acquisition_approved":False,
        "training_authorized":False,
        "disposition":("PROVISIONAL_preview_compatible_NOT_CLEARED"
                       if total and frac<=.01 and pair_count>0
                       else "HOLD_preview_tokenizer_or_continuation_failure"),
    }


def main():
    import boto3
    from botocore.config import Config
    client=boto3.client("s3",endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"))
    data=client.get_object(Bucket=os.environ["RUNPOD_NETWORK_VOLUME_ID"],Key=TOKENIZER_KEY)["Body"].read(10_000_000+1)
    if sha256(data).hexdigest()!=OLD_BYTES_SHA:
        raise ValueError("baseline frozen tokenizer was changed")
    tok=Tokenizer.from_str(data.decode())
    meta,metadata=get_json("https://huggingface.co/api/datasets/"+quote(DATASET,safe="/"),1_000_000)
    splits,splitmeta=get_json("https://datasets-server.huggingface.co/splits?"+
                              urlencode({"dataset":DATASET}),250_000)
    rows,rowmeta=get_json("https://datasets-server.huggingface.co/first-rows?"+
                          urlencode({"dataset":DATASET,"config":"default","split":"train"}),2_000_000)
    report=screen(meta,splits,rows,tok)
    report["hub_custody"]=metadata
    report["splits_custody"]=splitmeta
    report["first_rows_custody"]=rowmeta
    report["first_rows_at_exact_hub_revision"]=rowmeta.get("revision_header")==report["revision"]
    if not report["first_rows_at_exact_hub_revision"]:
        report["revision_limitation"]="viewer preview is not cryptographically pinned to the Hub revision"
    Path("tinystories_frozen_tokenizer_public_preview.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({key:report[key] for key in (
        "dataset_id","revision","preview_scanned_rows","raw_source_tokenizer",
        "continuation_window","first_rows_at_exact_hub_revision","disposition"
    )},sort_keys=True))


if __name__=="__main__":
    main()
