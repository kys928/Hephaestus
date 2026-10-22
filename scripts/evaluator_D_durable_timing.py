#!/usr/bin/env python3
"""Durable, stage-level evidence for a future separately approved D inference.

Every event is committed as a unique S3 object *before* entering the next
potentially blocking operation. A latest pointer is only a convenience; previous
events are immutable by name. No raw prompts, weights or credentials are stored.
This module never runs a pod, changes a checkpoint, or authorizes spending.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

import run_interface_repair_v1 as runtime

STAGES=frozenset({
    "runner_enter","source_result_fetch_start","source_result_fetch_done",
    "source_checkpoint_verified","curriculum_prepare_start","curriculum_prepare_done",
    "model_spec_verified","cache_resolution_start","hf_local_probe_start",
    "hf_local_probe_hit","hf_local_probe_miss","hf_ephemeral_download_start",
    "hf_ephemeral_download_done","hf_config_verified",
    "adapter_persistent_check_start","adapter_persistent_cache_hit",
    "adapter_ephemeral_download_start","adapter_ephemeral_download_done",
    "adapter_archive_verify_start","adapter_archive_verify_done",
    "adapter_extract_start","adapter_extract_done","adapter_ready",
    "cache_resolution_done","model_load_start","cuda_validation_start",
    "cuda_validation_done","tokenizer_load_start","tokenizer_load_done",
    "base_weights_load_start","base_weights_load_done",
    "lora_attach_start","lora_attach_done","model_eval_ready",
    "model_load_done","manifest_persist_start","manifest_persist_done",
    "inference_start","generation_started","generation_completed",
    "sample_shard_persisted","inference_complete","terminal_persisted",
    "runner_failed",
})

class DurableTrace:
    def __init__(self, client, prefix, run_id, repo_sha, *, monotonic=None, wall_clock=None):
        if not run_id.startswith("D-contrast-") or "/" in run_id or ".." in run_id:
            raise ValueError("trace run identity invalid")
        if len(repo_sha)!=40:
            raise ValueError("trace SHA invalid")
        self.client=client
        self.prefix=prefix.rstrip("/")+"/timing"
        self.run_id=run_id
        self.repo_sha=repo_sha
        self.monotonic=monotonic or time.monotonic
        self.wall_clock=wall_clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.started=self.monotonic()
        self.seq=0

    def __call__(self, stage, **detail):
        if stage not in STAGES:
            raise ValueError("unknown trace stage")
        # Intentionally small, closed metadata vocabulary. No provider response,
        # local storage path, exception text, generated content, or credentials.
        allowed={"cache_source","error_type","completed","total","shard_index",
                 "bytes","generated_tokens","elapsed_seconds"}
        if not set(detail)<=allowed:
            raise ValueError("unsupported trace metadata")
        data={}
        for key,value in detail.items():
            if value is not None and (type(value) not in (str,int,float,bool) or
                (type(value)==str and len(value)>80)):
                raise ValueError("invalid trace metadata")
            data[key]=value
        self.seq+=1
        record=dict(schema="D-contrast-stage-timing-v1",stage=stage,
                    sequence=self.seq,run_id=self.run_id,repo_sha=self.repo_sha,
                    utc=self.wall_clock(),elapsed_monotonic_seconds=round(
                        self.monotonic()-self.started,3),details=data)
        # Fail closed if evidence cannot be persisted; do not pretend to have a
        # measured stage from stdout only. These writes run only inside the
        # future approved pod's runner, not on this preparation branch.
        runtime.put_json(self.client,f"{self.prefix}/events/{self.seq:04d}.json",record)
        runtime.put_json(self.client,f"{self.prefix}/latest.json",record)
        print("D_CONTRAST_DURABLE_STAGE "+__import__("json").dumps(
            {k:record[k] for k in ("stage","sequence","elapsed_monotonic_seconds","details")},
            sort_keys=True),flush=True)
        return record
