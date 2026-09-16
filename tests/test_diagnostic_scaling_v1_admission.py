from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_diagnostic_scaling_v1.py"
spec = importlib.util.spec_from_file_location("diagnostic_scaling_preflight", SCRIPT)
assert spec and spec.loader
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def load_contract() -> dict:
    return json.loads((ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json").read_text())


def synthetic_training() -> bytes:
    rows = []
    for role in ("diagnosis", "controller"):
        for index in range(48):
            rows.append({"role": role, "index": index, "target": {"decision": "x"}})
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


def remote_rows(contract: dict) -> list[dict]:
    return [
        {
            "model_id": row["model_id"],
            "revision": row["revision"],
            "model_type": row["model_type"],
            "remote_revision_verified": True,
        }
        for row in contract["candidates"]
    ]


def test_build_admission_is_ephemeral_and_exactly_covers_roles():
    contract = copy.deepcopy(load_contract())
    training = synthetic_training()
    contract["sources"]["source_training_dataset_sha256"] = hashlib.sha256(training).hexdigest()
    admission = preflight.build_admission(
        repo_sha="a" * 40,
        contract=contract,
        contract_raw=json.dumps(contract).encode(),
        base_distance_raw=b"{}",
        training_raw=training,
        contamination_raw=json.dumps({"passed": True}).encode(),
        requirements_raw=preflight.requirements_bytes(contract),
        remote_rows=remote_rows(contract),
        runtime_compatibility={"peft_target_parameters_supported": True},
        source_training_key="scientific/source/training.jsonl",
    )
    assert admission["status"] == "launch_inputs_admitted"
    assert admission["roles"] == ["diagnosis", "controller"]
    assert admission["role_counts"] == {"diagnosis": 48, "controller": 48}
    assert admission["network_volume_attached"] is False
    assert admission["persistent_model_cache_allowed"] is False
    assert admission["promotion_allowed"] is False


def test_build_admission_rejects_incomplete_role_data():
    contract = copy.deepcopy(load_contract())
    training = synthetic_training().splitlines()
    training = b"\n".join(training[:-1]) + b"\n"
    contract["sources"]["source_training_dataset_sha256"] = hashlib.sha256(training).hexdigest()
    with pytest.raises(RuntimeError, match="role counts drifted"):
        preflight.build_admission(
            repo_sha="b" * 40,
            contract=contract,
            contract_raw=json.dumps(contract).encode(),
            base_distance_raw=b"{}",
            training_raw=training,
            contamination_raw=json.dumps({"passed": True}).encode(),
            requirements_raw=preflight.requirements_bytes(contract),
            remote_rows=remote_rows(contract),
            runtime_compatibility={},
            source_training_key="scientific/source/training.jsonl",
        )


def test_build_admission_rejects_unverified_candidate():
    contract = copy.deepcopy(load_contract())
    training = synthetic_training()
    contract["sources"]["source_training_dataset_sha256"] = hashlib.sha256(training).hexdigest()
    remote = remote_rows(contract)
    remote[2]["remote_revision_verified"] = False
    with pytest.raises(RuntimeError, match="remote model verification"):
        preflight.build_admission(
            repo_sha="c" * 40,
            contract=contract,
            contract_raw=json.dumps(contract).encode(),
            base_distance_raw=b"{}",
            training_raw=training,
            contamination_raw=json.dumps({"passed": True}).encode(),
            requirements_raw=preflight.requirements_bytes(contract),
            remote_rows=remote,
            runtime_compatibility={},
            source_training_key="scientific/source/training.jsonl",
        )
