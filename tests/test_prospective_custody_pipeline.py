"""Adversarial software tests. Stub generations below are never scientific evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hephaestus.evaluation import prospective_pipeline as pipeline
from hephaestus.evaluation.prospective_design import build_custodian_holdout, holdout_commitment, validate_holdout_distribution
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json


@pytest.fixture
def custody(tmp_path, monkeypatch):
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    design = ProspectiveDesign(purpose="rehearsal", cases_per_family=1, max_new_tokens=16)
    development = generate_original_splits(GeneratedDataSpec(20261028))[0]["development"][:4]
    exposure = {r["sample_id"] for r in development}
    rows = build_custodian_holdout(design, custodian_seed=152200, exposure_sample_ids=exposure)
    identity = "sha256:" + "1" * 64
    bindings = [ProspectiveModelBinding("baseline", "unused", identity, "unused", identity)]
    bindings += [ProspectiveModelBinding(f"seed-{seed}", "unused", identity, "unused", identity, seed)
                 for seed in design.training_seeds]
    calls = []

    def stub_generate(binding, items, _design):
        calls.append(binding.label)
        return {"model_identity": binding.identity(), "execution_profile": {"stub": True},
            "scorer_version": pipeline.SCORER_VERSION, "chat_template_sha256": identity,
            "samples": [{"sample_id": r["sample_id"], "prompt_tokens": [3, 4],
                "generated_tokens": [5, 1], "eos_token_id": 1, "eos": True, "truncated": False,
                "raw_decode": "invalid<eos>", "answer_text": "invalid"} for r in items]}

    monkeypatch.setattr(pipeline, "generate_cpu", stub_generate)
    monkeypatch.setattr(pipeline, "verify_model_binding", lambda *_: None)
    monkeypatch.setattr(pipeline, "generation_profile", lambda: {"stub": True})
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    repository = SQLiteStateRepository(tmp_path / "state.sqlite")
    selection = pipeline.freeze_development_selection(artifacts, design, development, bindings)
    calls.clear()
    submission = {"study_id": "software-unit-test", "design_commitment": design.commitment(),
        "execution_profile": {"stub": True}, "models": {b.label: b.identity() for b in bindings},
        "selection": selection, "holdout_commitment": holdout_commitment(rows),
        "exposure_commitment": pipeline.digest(sorted(exposure)), "scorer_version": pipeline.SCORER_VERSION}
    key = crypto.Ed25519PrivateKey.generate()
    now = datetime.now(timezone.utc)
    payload = {"study_id": "software-unit-test", "purpose": "rehearsal", "allowed_sessions": 1,
        "submission_commitment": pipeline.digest(submission), "authority_id": "unit-test-fixture",
        "issued_at": now.isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    grant = {"payload": payload, "signature_hex": key.sign(canonical_bytes(payload)).hex()}
    trust = {"unit-test-fixture": key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()}
    args = dict(study_id="software-unit-test", design=design, rows=rows,
                exposure_sample_ids=exposure, bindings=bindings, selection=selection, grant=grant)
    return pipeline.ProspectiveCustodyRunner(repository, artifacts, trust), args, calls


def test_one_session_restart_returns_exact_result_without_generation(custody):
    runner, args, calls = custody
    bundle = runner.run(**args)
    assert len(calls) == 4
    restarted = pipeline.ProspectiveCustodyRunner(runner.repository, runner.artifacts, runner.trusted_public_keys)
    assert restarted.run(**args) == bundle
    assert len(calls) == 4
    result = pipeline.verify_cold_bundle(bundle, runner.artifacts,
        expected_bundle_ref=runner.sessions.get(args["study_id"])["bundle_ref"],
        trusted_public_keys=runner.trusted_public_keys)
    assert result["verified"] and not result["token_decoding_verified"]
    assert not bundle["selected_procedure_effect_supported"]
    assert not bundle["external_certification"] and not bundle["model_promoted"]


@pytest.mark.parametrize("kind", ["wrong_key", "expired", "boolean_sessions", "changed_model"])
def test_invalid_authority_or_scope_never_exposes_holdout(custody, kind):
    runner, args, calls = custody
    if kind == "wrong_key":
        runner.trusted_public_keys["unit-test-fixture"] = "00" * 32
    elif kind == "expired":
        args["grant"]["payload"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    elif kind == "boolean_sessions":
        args["grant"]["payload"]["allowed_sessions"] = True
    else:
        args["bindings"][0].model_identity = "sha256:" + "2" * 64
    with pytest.raises((ValueError, PermissionError)):
        runner.run(**args)
    assert calls == []
    assert runner.sessions.get(args["study_id"]) is None


def test_failure_consumes_session_and_retains_partial_raw_outputs(custody, monkeypatch):
    runner, args, calls = custody
    original = pipeline.generate_cpu

    def interrupted(binding, rows, design):
        if binding.seed is not None:
            raise RuntimeError("predeclared interruption during final generation")
        return original(binding, rows, design)

    monkeypatch.setattr(pipeline, "generate_cpu", interrupted)
    with pytest.raises(RuntimeError):
        runner.run(**args)
    state = runner.sessions.get(args["study_id"])
    assert state["status"] == "failed" and set(state["partial_generation_refs"]) == {"baseline"}
    read_verified_json(runner.artifacts, state["partial_generation_refs"]["baseline"])
    monkeypatch.setattr(pipeline, "generate_cpu", original)
    with pytest.raises(PermissionError, match="consumed"):
        runner.run(**args)
    assert calls == ["baseline"]


def test_unknown_hard_stop_intent_is_not_replayed(custody):
    from hephaestus.schemas.prospective_custody import CustodySession
    runner, args, calls = custody
    commitment = args["grant"]["payload"]["submission_commitment"]
    runner.sessions.consume(CustodySession(args["study_id"], commitment, "intent-ref", "grant-ref",
                                           datetime.now(timezone.utc).isoformat()))
    with pytest.raises(PermissionError, match="consumed"):
        runner.run(**args)
    assert calls == []


def test_all_seed_attempts_and_actual_development_scores_are_required(custody):
    runner, args, calls = custody
    args["selection"]["attempts"][0]["development_correct"] = 1
    args["selection"]["attempts"][0]["development_valid_json"] = 1
    with pytest.raises(PermissionError, match="ledger differs"):
        runner.run(**args)
    with pytest.raises(ValueError, match="all predeclared"):
        pipeline.select_development_seed(args["selection"]["attempts"][:2], args["design"])
    assert calls == []


def test_cold_reanalysis_rejects_forged_eos_and_changed_raw_artifacts(custody):
    runner, args, _ = custody
    bundle = runner.run(**args)
    forged = deepcopy(bundle)
    output = read_verified_json(runner.artifacts, forged["generation_refs"]["seed-1117"])
    output["samples"][0]["eos"] = False
    forged["generation_refs"]["seed-1117"] = runner.artifacts.put_bytes(canonical_bytes(output)).artifact_ref
    with pytest.raises(ValueError, match="publication commitment"):
        pipeline.verify_cold_bundle(forged, runner.artifacts,
            expected_bundle_ref=runner.sessions.get(args["study_id"])["bundle_ref"],
            trusted_public_keys=runner.trusted_public_keys)
    with pytest.raises(ValueError, match="EOS/truncation"):
        pipeline.analyse_pairs(sorted(args["rows"], key=lambda r: r["sample_id"]), output, output, args["design"])
    forged = deepcopy(bundle)
    forged["primary"]["paired_exact"]["wins"] = 8
    with pytest.raises(ValueError, match="publication commitment"):
        pipeline.verify_cold_bundle(forged, runner.artifacts,
            expected_bundle_ref=runner.sessions.get(args["study_id"])["bundle_ref"],
            trusted_public_keys=runner.trusted_public_keys)


def test_correct_but_out_of_range_or_training_partition_item_is_rejected(custody):
    _, args, _ = custody
    row = {"family": "arithmetic", "prompt": "Compute 1000 plus 1. Return JSON with key result.",
           "target": '{"result":1001}'}
    from hephaestus.data.generated_validation import validate_original_example
    row["sample_id"] = pipeline.digest(validate_original_example(row))
    with pytest.raises(ValueError, match="partition|support"):
        validate_holdout_distribution([row], args["design"])


def test_actual_cpu_rehearsal_and_separate_cold_model_reexecution(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("cryptography")
    from hephaestus.development.prospective_rehearsal import run_rehearsal
    root = tmp_path / "actual-rehearsal"
    result = run_rehearsal(root)
    assert result["genuine_optimizer_updates"] == 12
    assert not result["confirmatory_effect_claimed"] and not result["external_certification"]
    replay = subprocess.run([sys.executable, "-m", "hephaestus.development.prospective_rehearsal", "cold", str(root)],
                            check=True, capture_output=True, text=True, timeout=120)
    cold = json.loads(replay.stdout)
    assert cold["verified"] and cold["model_reexecution_verified"]
    bindings = json.loads((root / "bindings.json").read_text())["bindings"]
    binding = ProspectiveModelBinding.from_dict(bindings[1])
    binding.adapter_path = bindings[2]["adapter_path"]
    binding.adapter_identity = bindings[2]["adapter_identity"]
    with pytest.raises(PermissionError, match="adapter/base/tokenizer"):
        pipeline.verify_model_binding(binding, ProspectiveDesign(purpose="rehearsal", cases_per_family=2, max_new_tokens=16))
