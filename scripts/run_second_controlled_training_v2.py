#!/usr/bin/env python3
"""Run the second controlled training with explicit DatasetManifest content-hash projection.

DatasetManifest stores the processed content hash on its single dataset entry.
The controlled driver wants a convenience top-level property for an invariant
check. This adapter exposes exactly that frozen value without mutating evidence.
"""
from __future__ import annotations

from hephaestus.schemas.dataset_manifest import DatasetManifest

import run_second_controlled_training as base


def _processed_content_hash(manifest: DatasetManifest) -> str | None:
    if len(manifest.datasets) != 1:
        raise RuntimeError("controlled dataset manifest must contain exactly one selected dataset entry")
    return str(manifest.datasets[0].get("content_hash") or "") or None


setattr(DatasetManifest, "processed_content_hash", property(_processed_content_hash))


if __name__ == "__main__":
    raise SystemExit(base.main())
