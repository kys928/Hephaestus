#!/usr/bin/env python3
"""One-source live Hugging Face *public preview* triage with original tokenizer.

Exact new source is research-only. No full-source acquisition, train, approvals,
protected refs or changes to the unmodified 1% unknown-token rule.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from tokenizers import Tokenizer

DATASET="x0root/math-hq-10k"
OLD_TOKENIZER="sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
OLD_BYTES_SHA="3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25"
TOKENIZER_KEY=("hephaestus/scientific/v1/runs/"
               "first-bounded-scientific-training-001-33866198758/"
               "checkpoint_step_100/tokenizer/tokenizer.json")
FIELDS={"instruction","input","output","problem_id"}
MAX_ROW_BYTES=24_000
MAX_PREVIEW_BYTES=2_000_000


def get_json(url:str,limit:int)->tuple[dict,dict]:
    request=Request(url,headers={"User-Agent":"hephaestus-one-source-preview-review/1"})
    with urlopen(request,timeout=24) as response:
        raw=response.read(limit+1)
        header={k.lower():v for k,v in response.headers.items()}
    if len(raw)>limit:
        raise ValueError("public source metadata exceeds read ceiling")
    return json.loads(raw),{"sha256":"sha256:"+sha256(raw).hexdigest(),
                            "response_bytes":len(raw),
                            "revision_header":header.get("x-revision")}


def preview_review(meta:dict,splits:dict,rows:dict,tokenizer:Tokenizer)->dict:
    revision=str(meta.get("sha",""))
    license_id=str(meta.get("cardData",{}).get("license","") or meta.get("license",""))
    if not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("immutable Hub commit identity missing")
    if license_id.lower()!="mit":
        raise ValueError("MIT metadata was not confirmed from Hub card")
    identifiers={(item.get("config"),item.get("split"))
                 for item in splits.get("splits",[])
                 if item.get("dataset")==DATASET}
    if ("default","train") not in identifiers:
        raise ValueError("expected train split missing")
    if rows.get("dataset")!=DATASET or rows.get("config")!="default" or rows.get("split")!="train":
        raise ValueError("source preview dataset/config/split mismatch")
    vocab=tokenizer.token_to_id("<|unk|>")
    if vocab is None:
        raise ValueError("original tokenizer unknown-token identity missing")
    stats=Counter()
    pairs=set()
    problems=set()
    unknown_raw=total_raw=unknown_flat=total_flat=0
    potential_pii=0
    for wrapper in rows.get("rows",[])[:100]:
        if wrapper.get("truncated_cells"):
            stats["viewer_truncated"]+=1
            continue
        row=wrapper.get("row",{})
        if not isinstance(row,dict) or not FIELDS<=set(row):
            stats["schema_mismatch"]+=1
            continue
        if any(type(row[field]) is not str or len(row[field].encode())>MAX_ROW_BYTES
               for field in FIELDS):
            stats["invalid_or_oversized_row"]+=1
            continue
        prompt=("Instruction: "+row["instruction"]+"\n"
                "Problem: "+row["input"])
        target=row["output"]
        if not prompt.strip() or not target.strip():
            stats["empty_pair"]+=1
            continue
        pair=(prompt.casefold().strip(),target.casefold().strip())
        if pair in pairs: stats["duplicate_pair"]+=1
        pairs.add(pair)
        if row["problem_id"] in problems:stats["repeated_problem_id"]+=1
        problems.add(row["problem_id"])
        raw=prompt+"\n"+target
        # Frozen/source-content style: newline is measured as-is, not hidden
        # by replacement. Flattened text only an explicitly separate control.
        ids=tokenizer.encode(raw).ids
        unknown_raw+=ids.count(vocab)
        total_raw+=len(ids)
        flattened=" ".join(raw.split())
        other=tokenizer.encode(flattened).ids
        unknown_flat+=other.count(vocab)
        total_flat+=len(other)
        potential_pii+=bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",raw))
        stats["valid_paired_preview"]+=1
        if "json" in prompt.casefold():stats["explicit_json_instruction"]+=1
        if "exactly" in prompt.casefold():stats["explicit_length_constraint"]+=1
        if any(ord(c)>127 for c in raw):stats["unicode_source_rows"]+=1
    fraction=unknown_raw/max(1,total_raw)
    return {
        "schema":"hephaestus-single-public-hf-candidate-screen.v1",
        "dataset_id":DATASET,"revision":revision,
        "license_metadata":license_id,"declared_source_origin":"author-published programmatic SymPy math",
        "declared_license_not_independent_chain_of_title":True,
        "description":"math reasoning and short-answer instructions; not an assessed structured JSON dataset",
        "preview_max_rows":100,"valid_rows":stats["valid_paired_preview"],
        "viewer_schema_fields":sorted(FIELDS),
        "preview_counts":dict(stats),
        "unique_normalized_pairs_in_preview":len(pairs),
        "unique_problem_ids_in_preview":len(problems),
        "split_rule":"group_by_problem_id_not_row_to_prevent_style_variant_leakage",
        "original_tokenizer_ref":OLD_TOKENIZER,
        "raw_newline_tokenizer":{"total_tokens":total_raw,"unk":unknown_raw,
                                 "unknown_fraction":fraction,
                                 "passes_unchanged_one_percent_ceiling":total_raw>0 and fraction<=0.01},
        "flattened_whitespace_control_only":{"total_tokens":total_flat,"unk":unknown_flat,
                                              "unknown_fraction":unknown_flat/max(1,total_flat),
                                              "not_used_to_pass_gate":True},
        "preview_possible_email_pattern_rows":potential_pii,
        "no_full_corpus_privacy_clearance":True,
        "no_protected_reference_leakage_clearance":True,
        "no_full_corpus_dedup":True,
        "original_workflow_dataset_selection_performed":False,
        "acquisition_approval":False,"training_authorized":False,
        "recommendation":(
            "HOLD_raw_tokenizer_above_one_percent" if total_raw and fraction>.01
            else "HOLD_requires_full_source_provenance_dedup_and_private_eval_clearance"
        ),
    }


def main():
    import boto3
    from botocore.config import Config
    client=boto3.client("s3",
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"))
    raw=client.get_object(Bucket=os.environ["RUNPOD_NETWORK_VOLUME_ID"],
                          Key=TOKENIZER_KEY)["Body"].read(10_000_000+1)
    if sha256(raw).hexdigest()!=OLD_BYTES_SHA:
        raise ValueError("original frozen tokenizer S3 bytes did not match")
    tokenizer=Tokenizer.from_str(raw.decode("utf-8"))
    meta,meta_http=get_json(
        "https://huggingface.co/api/datasets/"+quote(DATASET,safe="/"),1_000_000)
    splits,split_http=get_json(
        "https://datasets-server.huggingface.co/splits?"+urlencode({"dataset":DATASET}),200_000)
    preview,preview_http=get_json(
        "https://datasets-server.huggingface.co/first-rows?"+
        urlencode({"dataset":DATASET,"config":"default","split":"train"}),MAX_PREVIEW_BYTES)
    decision=preview_review(meta,splits,preview,tokenizer)
    decision["source_read_scope"]="public_dataset_metadata_and_first_100_rows_only"
    decision["metadata_custody"]=meta_http
    decision["split_custody"]=split_http
    decision["preview_custody"]=preview_http
    decision["viewer_revision_matches_hub"]=(preview_http["revision_header"]==decision["revision"])
    if not decision["viewer_revision_matches_hub"]:
        decision["provenance_limitation"]="Viewer first-rows revision not proven identical to Hub immutable commit"
    out=Path("one_hf_source_public_preview_screen.json")
    out.write_text(json.dumps(decision,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        k:decision[k] for k in (
        "dataset_id","revision","license_metadata","valid_rows",
        "raw_newline_tokenizer","recommendation","viewer_revision_matches_hub")
    },sort_keys=True))


if __name__=="__main__":
    main()
