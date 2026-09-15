from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/hephaestus_adaptation_elasticity_v1.json"
TOPOLOGY_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"


def _load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_low_dose_and_baseline_hash() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    topology_raw = TOPOLOGY_PATH.read_bytes()
    assert protocol["protocol_id"] == "hephaestus_adaptation_elasticity_v1"
    assert protocol["baseline"]["protocol_sha256"] == hashlib.sha256(topology_raw).hexdigest()
    assert protocol["scientific_variable"] == "base_model_identity_under_fixed_role_lora_dose"
    assert protocol["roles"] == ["diagnosis", "planner", "evaluator", "judge", "controller"]
    train = protocol["training"]
    assert train["method"] == "lora"
    assert train["peft_version"] == "0.20.0"
    assert train["rank"] == 8 and train["alpha"] == 16 and train["dropout"] == 0.0
    assert train["learning_rate"] == 5e-5
    assert train["epochs"] == 2 and train["dose_checkpoints"] == [1, 2]
    assert train["micro_batch_size"] == 1 and train["gradient_accumulation_steps"] == 4
    assert protocol["governance"]["promotion_allowed"] is False
    assert protocol["governance"]["lineage_mutation_allowed"] is False
    assert protocol["governance"]["training_is_experimentally_approved"] is True


def test_dataset_is_balanced_and_does_not_contaminate_frozen_topology() -> None:
    builder = _load_script("elasticity_builder", "scripts/build_adaptation_elasticity_dataset_v1.py")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    topology = json.loads(TOPOLOGY_PATH.read_text())
    rows = builder.build_dataset()
    assert len(rows) == 240
    for role in protocol["roles"]:
        assert sum(row["role"] == role for row in rows) == 48
    report = builder.contamination_report(rows, topology, protocol)
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["observed_maxima"]["jaccard"] <= protocol["dataset"]["max_normalized_token_jaccard"]
    required = {"decision", "action", "primary_variable", "confidence", "evidence_refs", "uncertainties", "rationale"}
    for row in rows:
        assert set(row["target"]) == required
        assert set(row["target"]["evidence_refs"]).issubset(set(row["allowed_evidence_refs"]))
        assert 0.0 <= float(row["target"]["confidence"]) <= 1.0


def test_controller_training_examples_follow_registry_semantics() -> None:
    builder = _load_script("elasticity_builder_controller", "scripts/build_adaptation_elasticity_dataset_v1.py")
    from hephaestus.policy.action_registry import evaluate_action_boundary

    for row in builder.build_dataset():
        if row["role"] != "controller":
            continue
        action = row["target"]["action"]
        facts = " ".join(item["fact"] for item in row["evidence"])
        approval = ""
        if "override_approved" in facts:
            approval = "override_approved"
        elif "approved" in facts:
            approval = "approved"
        boundary = evaluate_action_boundary(action, {"approval_status": approval})
        expected = "allowed" if boundary["allowed"] else "blocked"
        assert row["target"]["decision"] == expected
        assert row["target"]["primary_variable"] == "action_boundary"


def test_lora_target_selector_excludes_visual_modules() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    runner = _load_script("elasticity_runner", "scripts/run_adaptation_elasticity_v1.py")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.q_proj = torch.nn.Linear(4, 4)
            self.language_model.k_proj = torch.nn.Linear(4, 4)
            self.vision_model = torch.nn.Module()
            self.vision_model.q_proj = torch.nn.Linear(4, 4)
            self.other = torch.nn.Linear(4, 4)

    targets = runner._target_modules(Tiny(), ["q_proj", "k_proj"])
    assert targets == ["language_model.k_proj", "language_model.q_proj"]


def test_elasticity_metric_uses_signed_gain_and_training_cost() -> None:
    runner = _load_script("elasticity_runner_metric", "scripts/run_adaptation_elasticity_v1.py")
    baseline = {"quality_100": 50.0, "schema_compliance": 0.8, "evidence_grounding": 0.7, "confidence_calibration": 0.5, "hallucination_rate": 0.1}
    adapted = {"quality_100": 60.0, "schema_compliance": 1.0, "evidence_grounding": 0.8, "confidence_calibration": 0.7, "hallucination_rate": 0.05}
    record = runner._elasticity_record(baseline=baseline, adapted=adapted, dose=1, training_seconds=1800.0, training_tokens=500_000, trainable_parameters=10_000_000, adapter_manifest={"adapter_bytes": 1234}, training_peak_vram_bytes=10, mean_loss=0.4)
    assert record["delta_quality_points"] == 10.0
    assert record["delta_per_gpu_hour"] == 20.0
    assert record["delta_per_million_trainable_parameters"] == 1.0
    assert record["delta_per_million_training_tokens"] == 20.0


def test_launcher_shell_uses_adaptation_admission_and_driver_when_s3_available() -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script("elasticity_launcher", "scripts/launch_adaptation_elasticity_v1.py")
    shell = launcher.pod_shell_elasticity()
    assert "/adaptation_elasticity/model_admission/$HEPHAESTUS_REPO_SHA" in shell
    assert '"$PY" scripts/run_adaptation_elasticity_v1.py' in shell
    assert "run_positive_promotion_proof_v5.py" not in shell


def _resume_fixture() -> tuple[dict, dict, dict, dict, dict]:
    candidate = {
        "model_id": "example/model",
        "revision": "0123456789abcdef",
        "license": "apache-2.0",
    }
    protocol = {
        "protocol_id": "hephaestus_adaptation_elasticity_v1",
        "training": {
            "peft_version": "0.20.0",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "bias": "none",
            "target_module_suffixes": ["q_proj", "v_proj"],
            "dose_checkpoints": [1, 2],
        },
    }
    topology = {
        "generation": {"seeds": [11]},
        "cases": [
            {
                "case_id": "judge-1",
                "role": "judge",
                "condition": "base",
                "pair_id": None,
            }
        ],
    }
    baseline = {
        "quality_100": 50.0,
        "schema_compliance": 0.5,
        "evidence_grounding": 0.5,
        "confidence_calibration": 0.5,
        "hallucination_rate": 0.5,
    }
    identity = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": "a" * 64,
        "topology_protocol_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "training_config_sha256": "d" * 64,
    }
    return candidate, protocol, topology, baseline, identity


def _fixture_score(_topology: dict, _case: dict, output: str) -> dict:
    return {
        "quality": 0.6,
        "quality_100": 60.0,
        "components": {
            "schema": 1.0,
            "decision": 1.0,
            "action": 1.0,
            "primary_variable": 1.0,
            "evidence_grounding": 0.75,
            "confidence_calibration": 0.75,
            "forbidden_claim_avoidance": 1.0,
        },
        "schema_compliant": True,
        "hallucination_rate": 0.25,
        "forbidden_claim_hits": [],
        "parsed": {"output": output},
    }


def _fixture_summary(samples: list[dict]) -> dict:
    assert len(samples) == 1
    return {
        "quality": 0.6,
        "quality_100": 60.0,
        "schema_compliance": 1.0,
        "evidence_grounding": 0.75,
        "confidence_calibration": 0.75,
        "hallucination_rate": 0.25,
        "exact_repeatability": 1.0,
        "decision_repeatability": 1.0,
        "mean_generated_tokens": 4.0,
        "mean_ttft_seconds": 0.1,
        "mean_tokens_per_second": 10.0,
        "mean_total_latency_seconds": 0.4,
        "peak_vram_bytes": 100,
        "robustness_to_irrelevant_evidence": 0.0,
        "contradictory_evidence_quality": None,
        "sample_count": 1,
        "case_count": 1,
    }


def _write_valid_dose(
    resume_module,
    location,
    *,
    candidate: dict,
    protocol: dict,
    baseline: dict,
    identity: dict,
    dose: int,
) -> None:
    role = "judge"
    adapter_dir = location.adapter_path(role, dose)
    adapter_dir.mkdir(parents=True)
    config = {
        "peft_type": "LORA",
        "peft_version": protocol["training"]["peft_version"],
        "r": protocol["training"]["rank"],
        "lora_alpha": protocol["training"]["alpha"],
        "lora_dropout": protocol["training"]["dropout"],
        "bias": protocol["training"]["bias"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
        "target_modules": protocol["training"]["target_module_suffixes"],
        "base_model_name_or_path": f"/cache/snapshots/{candidate['revision']}",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(f"weights-{dose}".encode())
    components = {
        name: "sha256:" + resume_module.sha256_file(adapter_dir / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    manifest = {
        "manifest_version": resume_module.CURRENT_MANIFEST_VERSION,
        "model_id": candidate["model_id"],
        "revision": candidate["revision"],
        "role": role,
        "dose_epoch": dose,
        "trainable_parameters": 1_000_000,
        "target_module_count": 2,
        "target_modules_sha256": "e" * 64,
        "components": components,
        "adapter_bytes": sum((adapter_dir / name).stat().st_size for name in components),
        "manifest_sha256": hashlib.sha256(
            json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **identity,
    }
    (adapter_dir / "hephaestus_adapter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sample = {
        "model_id": candidate["model_id"],
        "revision": candidate["revision"],
        "case_id": "judge-1",
        "role": role,
        "condition": "base",
        "pair_id": None,
        "seed": 11,
        "output": "valid",
        "generation": {"generated_tokens": 4},
        "score": _fixture_score({}, {}, "valid"),
    }
    sample_path = location.sample_path(role, dose)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    summary = _fixture_summary([sample])
    record = {
        "dose_epoch": dose,
        "baseline_quality_100": baseline["quality_100"],
        "adapted_quality_100": summary["quality_100"],
        "delta_quality_points": 10.0,
        "delta_per_gpu_hour": 100.0,
        "delta_per_million_trainable_parameters": 10.0,
        "delta_per_million_training_tokens": 1000.0,
        "schema_compliance_delta": 0.5,
        "evidence_grounding_delta": 0.25,
        "confidence_calibration_delta": 0.25,
        "hallucination_rate_delta": -0.25,
        "training_seconds_cumulative": 10.0 * dose,
        "gpu_hours_cumulative": 10.0 * dose / 3600,
        "training_tokens_cumulative": 10 * dose,
        "trainable_parameters": 1_000_000,
        "adapter_bytes": manifest["adapter_bytes"],
        "training_peak_vram_bytes": 100,
        "mean_training_loss_epoch": 0.5,
        "adapted_role_summary": summary,
        "adapter_manifest": manifest,
        "evidence_identity": identity,
    }
    result_path = location.result_path(role, dose)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record), encoding="utf-8")


def _inspect_fixture(tmp_path: Path, doses: tuple[int, ...] = ()):
    resume_module = _load_script(
        f"elasticity_resume_{len(doses)}_{tmp_path.name}",
        "scripts/adaptation_elasticity_resume_v1.py",
    )
    candidate, protocol, topology, baseline, identity = _resume_fixture()
    location = resume_module.standard_location(tmp_path, "example--model")
    for dose in doses:
        _write_valid_dose(
            resume_module,
            location,
            candidate=candidate,
            protocol=protocol,
            baseline=baseline,
            identity=identity,
            dose=dose,
        )
    selection = resume_module.choose_role_evidence(
        tmp_path,
        model_slug="example--model",
        candidate=candidate,
        role="judge",
        protocol=protocol,
        topology=topology,
        baseline_summary=baseline,
        identity=identity,
        score_response=_fixture_score,
        summarize_role=_fixture_summary,
    )
    return resume_module, location, selection, candidate, protocol, topology, baseline, identity


def test_workflow_propagates_launcher_failure_through_tee(tmp_path: Path) -> None:
    workflow = (ROOT / ".github/workflows/adaptation-elasticity-v1.yml").read_text()
    assert "set -o pipefail\n          python scripts/launch_adaptation_elasticity_v1.py | tee" in workflow
    completed = subprocess.run(
        ["bash", "-c", "set -o pipefail; (exit 17) | tee /dev/null"],
        text=True,
        capture_output=True,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == 17


def test_terminal_poll_retries_transient_s3_credential_service_failure(monkeypatch) -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script(
        "elasticity_launcher_transient_s3",
        "scripts/launch_adaptation_elasticity_v1.py",
    )
    calls = 0

    def maybe_read_key(_client, _key):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "AccessDenied: failed to fetch user keys: unexpected status 503"
            )
        return json.dumps({"status": "completed"}).encode()

    monkeypatch.setattr(launcher.launcher.base, "maybe_read_key", maybe_read_key)
    result, observations = launcher.wait_for_elasticity_result(
        object(),
        object(),
        proof_run_id="run-1",
        attempt=2,
        pod_id="pod-1",
        max_seconds=1,
        poll_seconds=0,
        sleep_fn=lambda _seconds: None,
    )
    assert result == {"status": "completed"}
    assert calls == 2
    assert observations[0]["status"] == "retryable_error"


def test_terminal_poll_does_not_mask_nontransient_s3_auth_failure(monkeypatch) -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script(
        "elasticity_launcher_nontransient_s3",
        "scripts/launch_adaptation_elasticity_v1.py",
    )

    def maybe_read_key(_client, _key):
        raise RuntimeError("AccessDenied: invalid access key")

    monkeypatch.setattr(launcher.launcher.base, "maybe_read_key", maybe_read_key)
    with pytest.raises(RuntimeError, match="invalid access key"):
        launcher.wait_for_elasticity_result(
            object(),
            object(),
            proof_run_id="run-1",
            attempt=2,
            pod_id="pod-1",
            max_seconds=1,
            poll_seconds=0,
            sleep_fn=lambda _seconds: None,
        )


def test_pod_teardown_retries_transient_delete_and_verifies_absence() -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script(
        "elasticity_launcher_teardown_retry",
        "scripts/launch_adaptation_elasticity_v1.py",
    )

    class Execution:
        def __init__(self) -> None:
            self.delete_calls = 0
            self.inventory_calls = 0

        def delete_pod(self, pod_id: str) -> None:
            assert pod_id == "pod-1"
            self.delete_calls += 1
            if self.delete_calls == 1:
                raise RuntimeError(
                    "RunPod API request failed with HTTP 500: "
                    '{"error":"delete pod: unexpected end of JSON input"}'
                )

        def _request(self, method: str, path: str):
            assert (method, path) == ("GET", "pods")
            self.inventory_calls += 1
            if self.inventory_calls == 1:
                return 200, [
                    {"id": "pod-1", "name": "hephaestus-run-1", "desiredStatus": "RUNNING"}
                ]
            return 200, []

    execution = Execution()
    record = launcher.delete_pod_with_retries(
        execution,
        "pod-1",
        attempts=3,
        delay_seconds=0,
        sleep_fn=lambda _seconds: None,
    )
    assert record["deleted"] is True
    assert record["verified_absent"] is True
    assert record["attempts"] == 2
    assert execution.delete_calls == 2


def test_prelaunch_cleanup_deletes_only_matching_experiment_pods(monkeypatch) -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script(
        "elasticity_launcher_scoped_cleanup",
        "scripts/launch_adaptation_elasticity_v1.py",
    )
    inventories = [
        [
            {"id": "target", "name": "hephaestus-run-1", "desiredStatus": "RUNNING"},
            {"id": "other", "name": "unrelated-pod", "desiredStatus": "RUNNING"},
        ],
        [{"id": "other", "name": "unrelated-pod", "desiredStatus": "RUNNING"}],
    ]
    deleted: list[str] = []

    monkeypatch.setattr(
        launcher,
        "list_pods_with_retries",
        lambda _execution, **_kwargs: inventories.pop(0),
    )

    def delete(_execution, pod_id: str):
        deleted.append(pod_id)
        return {"deleted": True, "verified_absent": True}

    monkeypatch.setattr(launcher, "delete_pod_with_retries", delete)
    record = launcher.cleanup_experiment_pods(object(), "hephaestus-run-1")
    assert deleted == ["target"]
    assert record["matched_pods"] == ["target"]
    assert record["dangling_pods"] == []


def test_existing_completed_attempt_is_reused_only_after_full_verification(monkeypatch) -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script(
        "elasticity_launcher_existing_terminal",
        "scripts/launch_adaptation_elasticity_v1.py",
    )
    reads: list[str] = []

    def maybe_read_key(_client, key: str):
        reads.append(key)
        if key.endswith("adaptation_elasticity/run-1/elasticity_result.json"):
            return None
        if key.endswith("attempt-2/driver_result.json"):
            return json.dumps({"status": "completed", "marker": "attempt-2"}).encode()
        return json.dumps({"status": "failed"}).encode()

    def verify(payload: dict, _protocol: dict):
        if payload.get("status") != "completed":
            raise RuntimeError("not complete")
        return {"role_rankings": {"judge": []}}

    monkeypatch.setattr(launcher.launcher.base, "maybe_read_key", maybe_read_key)
    monkeypatch.setattr(launcher, "_verify", verify)
    observations: list[dict] = []
    found = launcher._find_completed_result(
        object(),
        proof_run_id="run-1",
        next_attempt=3,
        protocol={},
        observations=observations,
    )
    assert found is not None
    payload, verification, key = found
    assert payload["marker"] == "attempt-2"
    assert verification == {"role_rankings": {"judge": []}}
    assert key.endswith("attempt-2/driver_result.json")
    assert len(reads) == 2
    assert observations[-1]["status"] == "verified"


def test_complete_role_is_detected_and_skipped(tmp_path: Path) -> None:
    resume_module, _, selection, *_ = _inspect_fixture(tmp_path, (1, 2))
    assert selection["state"] == "complete"
    assert selection["selected"]["valid_doses"] == [1, 2]
    assert resume_module.role_action(selection)["action"] == "reuse"


def test_missing_role_runs_from_base(tmp_path: Path) -> None:
    resume_module, _, selection, *_ = _inspect_fixture(tmp_path)
    assert selection["state"] == "missing"
    assert resume_module.role_action(selection)["action"] == "run_from_base"


def test_partial_role_reconstructs_from_base(tmp_path: Path) -> None:
    resume_module, _, selection, *_ = _inspect_fixture(tmp_path, (1,))
    assert selection["state"] == "partial"
    assert selection["standard"]["valid_doses"] == [1]
    assert resume_module.role_action(selection)["action"] == "reconstruct_from_base"


def test_corrupt_persisted_evidence_is_rejected(tmp_path: Path) -> None:
    resume_module, location, _, candidate, protocol, topology, baseline, identity = _inspect_fixture(tmp_path, (1, 2))
    location.sample_path("judge", 2).write_text("{", encoding="utf-8")
    selection = resume_module.choose_role_evidence(
        tmp_path,
        model_slug="example--model",
        candidate=candidate,
        role="judge",
        protocol=protocol,
        topology=topology,
        baseline_summary=baseline,
        identity=identity,
        score_response=_fixture_score,
        summarize_role=_fixture_summary,
    )
    assert selection["state"] == "invalid"
    assert resume_module.role_action(selection)["action"] == "reconstruct_from_base"


def test_protocol_mismatch_rejects_persisted_evidence(tmp_path: Path) -> None:
    resume_module, _, _, candidate, protocol, topology, baseline, identity = _inspect_fixture(tmp_path, (1, 2))
    changed_identity = {**identity, "protocol_sha256": "f" * 64}
    selection = resume_module.choose_role_evidence(
        tmp_path,
        model_slug="example--model",
        candidate=candidate,
        role="judge",
        protocol=protocol,
        topology=topology,
        baseline_summary=baseline,
        identity=changed_identity,
        score_response=_fixture_score,
        summarize_role=_fixture_summary,
    )
    assert selection["state"] == "invalid"
    assert any(
        "protocol_sha256" in error
        for dose in selection["standard"]["doses"]
        for error in dose["errors"]
    )


def test_peft_qualified_language_targets_normalize_to_frozen_suffixes(tmp_path: Path) -> None:
    resume_module = _load_script(
        f"elasticity_resume_qualified_targets_{tmp_path.name}",
        "scripts/adaptation_elasticity_resume_v1.py",
    )
    errors: list[str] = []
    resume_module._validate_configured_targets(
        [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        errors,
    )
    assert errors == []


def test_peft_qualified_visual_target_is_still_rejected(tmp_path: Path) -> None:
    resume_module = _load_script(
        f"elasticity_resume_visual_targets_{tmp_path.name}",
        "scripts/adaptation_elasticity_resume_v1.py",
    )
    errors: list[str] = []
    resume_module._validate_configured_targets(
        ["vision_tower.self_attn.q_proj", "language_model.self_attn.k_proj"],
        ["q_proj", "k_proj"],
        errors,
    )
    assert any("forbidden visual path" in error for error in errors)


def test_qwen_judge_partial_recovery_uses_separate_reconstruction(tmp_path: Path) -> None:
    resume_module, location, selection, *_ = _inspect_fixture(tmp_path, (1,))
    action = resume_module.role_action(selection)
    output = resume_module.output_location_for_action(
        tmp_path,
        model_slug="Qwen--Qwen3-4B-Instruct-2507",
        role="judge",
        attempt=2,
        action=action["action"],
    )
    assert output.kind == "reconstruction"
    assert "reconstructions/Qwen--Qwen3-4B-Instruct-2507/judge/attempt-2" in str(output.role_results_dir)
    assert location.result_path("judge", 1).is_file()


def test_completed_reconstruction_is_reused_on_next_attempt(tmp_path: Path) -> None:
    resume_module, _, _, candidate, protocol, topology, baseline, identity = _inspect_fixture(tmp_path, (1,))
    reconstructed = resume_module.reconstruction_location(tmp_path, "example--model", "judge", 2)
    for dose in (1, 2):
        _write_valid_dose(
            resume_module,
            reconstructed,
            candidate=candidate,
            protocol=protocol,
            baseline=baseline,
            identity=identity,
            dose=dose,
        )
    selection = resume_module.choose_role_evidence(
        tmp_path,
        model_slug="example--model",
        candidate=candidate,
        role="judge",
        protocol=protocol,
        topology=topology,
        baseline_summary=baseline,
        identity=identity,
        score_response=_fixture_score,
        summarize_role=_fixture_summary,
    )
    assert selection["state"] == "complete"
    assert selection["selected"]["location"]["kind"] == "reconstruction"
    assert resume_module.role_action(selection)["action"] == "reuse"


def test_valid_standard_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    resume_module, _, _, *_ = _inspect_fixture(tmp_path, (1, 2))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        resume_module.output_location_for_action(
            tmp_path,
            model_slug="example--model",
            role="judge",
            attempt=2,
            action="run_from_base",
        )


def test_final_verifier_requires_four_models_five_roles_and_two_doses() -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script("elasticity_launcher_final_validation", "scripts/launch_adaptation_elasticity_v1.py")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    topology = json.loads(TOPOLOGY_PATH.read_text())
    builder = _load_script("elasticity_builder_final_validation", "scripts/build_adaptation_elasticity_dataset_v1.py")
    dataset_raw = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in builder.build_dataset()
    ).encode()
    result = {
        "status": "completed",
        "disposition": "scientific_elasticity_complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "topology_protocol_sha256": hashlib.sha256(TOPOLOGY_PATH.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "contamination_status": "passed",
        "baseline_run_id": protocol["baseline"]["run_id"],
        "model_results": [
            {
                "model_id": candidate["model_id"],
                "revision": candidate["revision"],
                "status": "complete",
                "roles": {
                    role: {"doses": [{"dose_epoch": 1}, {"dose_epoch": 2}]}
                    for role in protocol["roles"]
                },
            }
            for candidate in topology["candidates"]
        ],
        "training_performed": True,
        "promotion_performed": False,
        "lineage_mutated": False,
    }
    launcher._verify(result, protocol)
    result["model_results"][3]["roles"]["controller"]["doses"].pop()
    with pytest.raises(RuntimeError, match="dose evidence is incomplete"):
        launcher._verify(result, protocol)
