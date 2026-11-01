"""Actual bounded CPU rehearsal; models, data and custody authority are fixtures.

No downloads, protected final holdout, paid services or external certificate.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.data.chunking import chunk_records
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingControlRequest
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes
from hephaestus.training.cpu_chat_json_recipe import chat_prompt
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService
from hephaestus.evaluation.prospective_design import build_custodian_holdout, holdout_commitment
from hephaestus.evaluation.prospective_pipeline import (
    ProspectiveCustodyRunner, digest, file_digest, freeze_development_selection, verify_cold_bundle,
    generation_profile,
)


def create_tiny_model(path: Path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast
    path.mkdir(parents=True, exist_ok=False)
    vocab = {t: i for i, t in enumerate(["<pad>", "<eos>", "<unk>", "<bos>", *sorted(pre_tokenizers.ByteLevel.alphabet())])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>", bos_token="<bos>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    torch.manual_seed(152)
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=len(tokenizer), n_positions=512,
        n_ctx=512, n_embd=32, n_layer=1, n_head=2, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
        eos_token_id=1, pad_token_id=0, bos_token_id=3))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return tokenizer, sum(p.numel() for p in model.parameters())


def prepare_cpu_proposal(root, *, run_id, seed, model_path, tokenizer, parameter_count, rows, steps=4):
    root.mkdir(parents=True, exist_ok=False)
    template = chat_prompt(tokenizer, "{prompt}") + "{target}"
    records = chunk_records(tuple({"prompt": r["prompt"], "target": r["target"] + tokenizer.eos_token} for r in rows),
        chunk_size_tokens=512, min_tokens=1, prompt_target_template=template).records
    if any(tokenizer.unk_token_id in tokenizer.encode(r["text"], add_special_tokens=False) for r in records):
        raise ValueError("tiny fixture tokenizer has unknown source tokens")
    data_path = root / "trainable.jsonl"
    data_path.write_bytes(b"".join(canonical_bytes(r)+b"\n" for r in records))
    contract_path = root / "contract.json"
    contract_path.write_bytes(canonical_bytes(TrainableDataContract("fixture-"+run_id, run_id,
        "fixture-generated-data", str(data_path), "trainable-data.v1", 1).to_dict()))
    evidence_path = root / "processing.json"
    evidence_path.write_bytes(canonical_bytes({"processed_dataset_ref": str(data_path),
        "processed_content_hash": file_digest(data_path),
        "wrapper": {"kind": "explicit_prompt_target_template", "template": template},
        "prompt_target_boundary": {"status": "explicit", "schema_version": "prompt-target-structured.v1"},
        "tokenizer_compatibility": {"status": "checked", "compatible": True,
            "tokenizer_ref": str(model_path), "checker_id": "actual-byte-tokenizer-no-unknown-v1"}}))
    revision = directory_content_identity(model_path)
    return ExperimentProposal(experiment_id="fixture-experiment-"+run_id, run_id=run_id,
        lineage_id="fixture-prospective-L", stage_name="smoke_test", diagnosis_report_id="fixture-rehearsal",
        intervention_id="fixture-seed-"+str(seed), primary_variable="training_seed", baseline_ref=revision, status="ready",
        training_constraints={"backend_id": "transformers_causal_lm", "model_id": str(model_path),
            "model_revision": revision, "tokenizer_id": str(model_path), "tokenizer_revision": revision,
            "tokenizer_ref": str(model_path), "architecture_family": "gpt2", "training_mode": "lora_finetune",
            "peft_config": {"rank": 2, "alpha": 4., "dropout": .1, "target_modules": ["c_attn"]},
            "trainable_data_contract_ref": str(contract_path), "trainable_data_contract_hash": file_digest(contract_path),
            "processed_dataset_ref": str(data_path), "processed_dataset_hash": file_digest(data_path),
            "processing_evidence_ref": str(evidence_path), "processing_evidence_hash": file_digest(evidence_path),
            "optimizer": "adamw", "scheduler": "constant", "parameter_count": parameter_count,
            "vocabulary_size": len(tokenizer), "hidden_size": 32,
            "special_token_ids": {"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2, "bos_token_id": 3},
            "context_length": 512, "device": "cpu", "dtype": "float32", "seed": seed,
            "learning_rate": .01, "batch_size": 1, "gradient_accumulation_steps": 1,
            "max_steps": steps, "checkpoint_every_steps": steps, "logging_every_steps": 1,
            "max_total_tokens": 100000, "local_files_only": True})


def run_rehearsal(root: Path, *, campaign_witness=False, external_fixture_root=None,
                  before_assessment=None):
    import torch
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    torch.set_num_threads(1)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model_path = root / "tiny-base"
    tokenizer, parameters = create_tiny_model(model_path)
    identity = directory_content_identity(model_path)
    splits, _ = generate_original_splits(GeneratedDataSpec(20261028))
    by_family = {f: [r for r in splits["train"] if r["family"] == f] for f in ("arithmetic", "json", "symbolic", "reasoning")}
    train = [by_family[f][i] for i in range(2) for f in by_family]
    development = [next(r for r in splits["development"] if r["family"] == f) for f in by_family]
    design = ProspectiveDesign(purpose="rehearsal", cases_per_family=2, max_new_tokens=16)
    service = TransformersTrainingLifecycleService(root / "training")
    bindings = [ProspectiveModelBinding("baseline", str(model_path), identity, str(model_path), identity)]
    for seed in design.training_seeds:
        run_id = f"rehearsal-{seed}"
        proposal = prepare_cpu_proposal(root / f"inputs-{seed}", run_id=run_id, seed=seed,
            model_path=model_path, tokenizer=tokenizer, parameter_count=parameters, rows=train)
        result = service.launch(proposal)
        deadline = time.monotonic() + 120
        while result.status in {"preparing", "queued", "running", "resuming"} and time.monotonic() < deadline:
            time.sleep(.05)
            result = service.status(run_id)
        if result.status != "completed":
            service.control(TrainingControlRequest("cancel-"+run_id, run_id, "cancel", "fixture-rehearsal", "bounded CPU deadline"))
            raise RuntimeError(f"actual CPU rehearsal training failed: {result.status}")
        checkpoint = Path(result.checkpoint_refs[0])
        job_path = root / "training" / run_id / "prepared_job.json"
        if not job_path.exists():
            job_path = root / "training" / run_id / "job.json"
        bindings.append(ProspectiveModelBinding(f"seed-{seed}", str(model_path), identity,
            str(model_path), identity, seed, str(checkpoint / "model"), directory_content_identity(checkpoint / "model"),
            str(checkpoint), result.metadata["checkpoint_manifest_hashes"][str(checkpoint)], str(job_path), file_digest(job_path)))
    artifacts = FileSystemArtifactStore(root / "custody-artifacts")
    repository = SQLiteStateRepository(root / "custody.sqlite3",campaign_witness=campaign_witness)
    if campaign_witness:
        from hephaestus.schemas.checkpoint_identity import CheckpointIdentity
        from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
        for binding in bindings[1:]:
            evidence=artifacts.put_bytes(canonical_bytes(binding.to_dict())).artifact_ref
            CheckpointIdentityStore(repository).retain(CheckpointIdentity('fixture-'+str(binding.seed),
                binding.checkpoint_path,binding.checkpoint_manifest_hash,evidence))
    selection = freeze_development_selection(artifacts, design, development, bindings)
    exposure = {r["sample_id"] for split in splits.values() for r in split}
    rows = build_custodian_holdout(design, custodian_seed=152100, exposure_sample_ids=exposure)
    submission = {"study_id": "cpu-fixture-rehearsal", "design_commitment": design.commitment(),
        "execution_profile": generation_profile(),
        "models": {b.label: b.identity() for b in bindings}, "selection": selection,
        "holdout_commitment": holdout_commitment(rows), "exposure_commitment": digest(sorted(exposure)),
        "scorer_version": "strict-json-object-finite-typed.v1"}
    # Fixture key only. Production code has no key-creation or signing path.
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    authority = "fixture-rehearsal-only-not-independent"
    now = datetime.now(timezone.utc)
    payload = {"study_id": "cpu-fixture-rehearsal", "purpose": "rehearsal", "allowed_sessions": 1,
        "submission_commitment": digest(submission), "authority_id": authority,
        "issued_at": now.isoformat(), "expires_at": (now+timedelta(hours=1)).isoformat()}
    grant = {"payload": payload, "signature_hex": key.sign(canonical_bytes(payload)).hex()}
    ledger_options={}
    if external_fixture_root is not None:
        from hephaestus.development.fixture_exposure_ledger import FixtureExposureLedger
        external_fixture_root=Path(external_fixture_root).absolute()
        if external_fixture_root.is_relative_to(root) or root.is_relative_to(external_fixture_root):
            raise ValueError('external fixture service must remain outside the campaign/backup root')
        ledger_key=Ed25519PrivateKey.generate()
        ledger_public=ledger_key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex()
        ledger=FixtureExposureLedger(SQLiteStateRepository(external_fixture_root/'ledger.sqlite3',campaign_witness=True),
            private_key=ledger_key,authority_id='fixture-exposure-only',
            trusted_custodian_keys={authority:public},namespace='fixture/public-launch-dry-run')
        ledger_options=dict(exposure_ledger=ledger,trusted_exposure_keys={'fixture-exposure-only':ledger_public},
                            trusted_ledger_namespace=ledger.namespace)
        trust={'keys':{'fixture-exposure-only':ledger_public},'namespace':ledger.namespace}
        (root/'fixture-exposure-trust.json').write_bytes(canonical_bytes(trust))
        (external_fixture_root/'fixture-exposure-trust.json').write_bytes(canonical_bytes(trust))
    if before_assessment is not None:
        before_assessment(repository,artifacts,ledger_options)
    bundle = ProspectiveCustodyRunner(repository, artifacts, {authority: public},**ledger_options).run(
        study_id="cpu-fixture-rehearsal", design=design, rows=rows, exposure_sample_ids=exposure,
        bindings=bindings, selection=selection, grant=grant)
    (root / "bundle.json").write_bytes(canonical_bytes(bundle))
    (root / "bindings.json").write_bytes(canonical_bytes({"bindings": [b.to_dict() for b in bindings]}))
    (root / "fixture-public-trust.json").write_bytes(canonical_bytes({authority: public}))
    return {"purpose": "rehearsal", "training_seeds": list(design.training_seeds),
        "genuine_optimizer_updates": 12, "final_examples": len(rows), "selected_seed": selection["selected_seed"],
        "primary": bundle["primary"], "confirmatory_effect_claimed": False, "external_certification": False}


def cold_rehearsal(root, *, reexecute=True, expected_bundle_ref=None, external_trust_file=None):
    root = Path(root)
    from hephaestus.state.prospective_session_store import ProspectiveSessionStore
    session = ProspectiveSessionStore(SQLiteStateRepository(root / "custody.sqlite3",read_only=True)).get("cpu-fixture-rehearsal")
    if expected_bundle_ref is not None and session['bundle_ref']!=expected_bundle_ref:
        raise ValueError('cold rehearsal publication differs from separately retained commitment')
    exposure={}
    trust_path=Path(external_trust_file) if external_trust_file is not None else root/'fixture-exposure-trust.json'
    if trust_path.exists():
        trust=json.loads(trust_path.read_text());exposure=dict(trusted_exposure_keys=trust['keys'],trusted_ledger_namespace=trust['namespace'])
    return verify_cold_bundle(json.loads((root / "bundle.json").read_text()),
        FileSystemArtifactStore(root / "custody-artifacts"),
        expected_bundle_ref=session["bundle_ref"],
        trusted_public_keys=json.loads((root / "fixture-public-trust.json").read_text()),
        bindings=[ProspectiveModelBinding.from_dict(v) for v in json.loads((root / "bindings.json").read_text())["bindings"]],
        reexecute=reexecute,**exposure)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "cold"))
    parser.add_argument("root", type=Path)
    parser.add_argument('--expected-bundle-ref')
    parser.add_argument('--external-trust-file',type=Path)
    args = parser.parse_args()
    print(json.dumps(run_rehearsal(args.root) if args.action == "run" else cold_rehearsal(args.root,
        expected_bundle_ref=args.expected_bundle_ref,external_trust_file=args.external_trust_file), indent=2))
