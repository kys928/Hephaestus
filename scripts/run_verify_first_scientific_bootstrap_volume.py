#!/usr/bin/env python3
"""Run the independent volume verifier while accepting any valid JSON evidence shape."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_verifier() -> ModuleType:
    path = Path(__file__).with_name("verify_first_scientific_bootstrap_volume.py")
    spec = importlib.util.spec_from_file_location("hephaestus_first_scientific_volume_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load volume verifier from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    verifier = _load_verifier()

    def _json_verified(client: Any, bucket: str, record: dict[str, object]) -> object:
        artifact_ref = str(record["artifact_ref"])
        content_hash = str(record.get("content_hash") or artifact_ref)
        if artifact_ref != content_hash:
            raise RuntimeError("evidence record artifact_ref/content_hash disagreement")
        key = verifier._key_from_s3(str(record["storage_path"]), bucket)
        if key != verifier._object_key(artifact_ref):
            raise RuntimeError("evidence storage path is not the content-addressed key")
        raw = verifier._read_verified(
            client,
            bucket,
            key,
            artifact_ref,
            expected_size=int(record["byte_size"]) if record.get("byte_size") is not None else None,
        )
        return json.loads(raw.decode("utf-8"))

    verifier._json_verified = _json_verified
    verifier.main()


if __name__ == "__main__":
    main()
