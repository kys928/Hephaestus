#!/usr/bin/env python3
"""Prepare the selected shard with an explicit JSON-extension/JSONL-content adapter.

The immutable selected file is named ``train/sql.json`` but its bytes contain
newline-delimited JSON objects. This integration adapter does not mutate or
rename the source. It catches only the JSON decoder's ``Extra data`` condition,
re-decodes the same immutable bytes through the existing JSONL loader, and
records the effective format in processing evidence.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import run_first_selected_dataset_preparation as base
import run_first_selected_dataset_preparation_v3  # noqa: F401 - installs RunPod byte checks
import hephaestus.data.preprocessing as preprocessing

_original_load_records = preprocessing._load_records
_original_process_remote = preprocessing.AutonomousDataPreprocessor.process_remote_acquisition
_format_fallback_used = False


def _load_records_with_explicit_jsonl_fallback(
    path: Path,
    record_format: str,
    max_rows: int,
):
    global _format_fallback_used
    if record_format != "json":
        return _original_load_records(path, record_format, max_rows)
    try:
        return _original_load_records(path, record_format, max_rows)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        _format_fallback_used = True
        print(
            "source_format_adapter "
            f"path={path} declared_format=json effective_format=jsonl "
            "reason=json_decoder_extra_data"
        )
        return _original_load_records(path, "jsonl", max_rows)


def _process_remote_with_format_evidence(self, *args, **kwargs):
    global _format_fallback_used
    _format_fallback_used = False
    result = _original_process_remote(self, *args, **kwargs)
    evidence = copy.deepcopy(result.processing_evidence)
    if _format_fallback_used:
        evidence["source_format"] = {
            "declared_record_format": "json",
            "effective_record_format": "jsonl",
            "detection": "json_decoder_extra_data",
            "source_bytes_mutated": False,
        }
        preprocessing_section = evidence.get("preprocessing")
        if isinstance(preprocessing_section, dict):
            operations = list(preprocessing_section.get("operations") or [])
            if "jsonl_decode_from_json_extension" not in operations:
                operations.insert(0, "jsonl_decode_from_json_extension")
            preprocessing_section["operations"] = operations
    return replace(result, processing_evidence=evidence)


preprocessing._load_records = _load_records_with_explicit_jsonl_fallback
preprocessing.AutonomousDataPreprocessor.process_remote_acquisition = _process_remote_with_format_evidence


if __name__ == "__main__":
    base.main()
