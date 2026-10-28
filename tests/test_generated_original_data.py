from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import pytest
from hephaestus.data.generated_factory import (generate_original_splits, prepare_original_source,
    generator_code_sha256, materialize_authorized_tokens, load_authorized_tokens)
from hephaestus.data.generated_validation import validate_original_example, OriginalTaskSemanticChecker
from hephaestus.data.internal_screening import InternalSourceScanner
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.schemas.internal_research import InternalResearchPolicy
from hephaestus.policy.internal_research import InternalDatasetAdmissionPolicy, put_record
from hephaestus.state.prospective_evaluation import ProspectiveEvaluationStore
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def test_original_generation_is_balanced_reproducible_and_semantically_disjoint():
    spec = GeneratedDataSpec(31, 12, 3, 3)
    splits, report = generate_original_splits(spec)
    assert generate_original_splits(spec) == (splits, report)
    sets = {name: {validate_original_example(row) for row in rows} for name, rows in splits.items()}
    assert not sets["train"] & (sets["development"] | sets["evaluation"])
    assert not sets["development"] & sets["evaluation"]
    assert all(len(v) == 12 for v in [splits["development"], splits["evaluation"]])
    for split in splits.values():
        for row in split:
            broken = {**row, "target": '{"result":true}'}
            with pytest.raises(ValueError):
                validate_original_example(broken)
    another, _ = generate_original_splits(replace(spec, seed=93))
    assert not sets["train"] & {validate_original_example(r) for r in another["evaluation"]}
    assert set(report["family_counts"]["train"].values()) == {12}


def create_governed_source(root, tokenizer_path):
    artifacts = FileSystemArtifactStore(root / "artifacts")
    repository = SQLiteStateRepository(root / "state.sqlite")
    evaluation = ProspectiveEvaluationStore(repository, artifacts)
    digest = "sha256:" + sha256(tokenizer_path.read_bytes()).hexdigest()
    scanner = InternalSourceScanner(artifacts, tokenizer_path, digest, OriginalTaskSemanticChecker())
    profile = "sha256:" + sha256(b'original-prompt-Answer-boundary-no-eos-no-truncation-v1').hexdigest()
    candidate, review, protocol, generator = prepare_original_source(artifacts, evaluation, scanner,
        GeneratedDataSpec(413, 12, 3, 3), experiment_id="original-repair", processing_profile_sha256=profile)
    policy = InternalResearchPolicy("original-data-cpu", "user-implementation-mandate", "original-repair",
        "cpu-research-workflow", "cpu-research-run", protocol, profile, digest,
        8*1024*1024, 2000, 2, 600, ("cc0-1.0",),
        allowed_generator_code_sha256=(generator_code_sha256(),),
        allowed_capabilities=("arithmetic", "json", "symbolic", "reasoning"))
    ref = put_record(artifacts, policy.to_dict())
    admission = InternalDatasetAdmissionPolicy(artifacts, evaluation, ref, frozenset({ref}), scanner)
    scope = {"experiment_id": policy.experiment_id, "workflow_id": policy.workflow_id,
        "run_id": policy.run_id, "processing_profile_sha256": profile, "tokenizer_sha256": digest}
    return admission, candidate, review, scope


def test_actual_generated_token_artifact_rechecks_source_and_exact_masks(tmp_path):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["original grammar"], trainers.BpeTrainer(vocab_size=300,
        special_tokens=["<unk>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    admission, candidate, review, scope = create_governed_source(tmp_path, path)
    material = materialize_authorized_tokens(admission, candidate, review, **scope)
    rows = load_authorized_tokens(admission, candidate, review, material, **scope)
    assert len(rows) == 48
    assert all(r["labels"] == [-100] * r["prefix_tokens"] + r["input_ids"][r["prefix_tokens"]:] for r in rows)
    with pytest.raises(PermissionError):
        load_authorized_tokens(admission, candidate, review, "sha256:" + "a" * 64, **scope)
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_authorized_tokens(admission, candidate, review, material, **scope)


def test_canonical_commutative_leakage_and_invalid_outputs_are_detected():
    a = {"prompt": "Compute 12 plus 7. Return JSON with key result.", "target": '{"result":19}', "family": "arithmetic"}
    b = {"prompt": "Compute 7 plus 12. Return JSON with key result.", "target": '{"result":19}', "family": "arithmetic"}
    assert OriginalTaskSemanticChecker().check([a], [b])["overlap"] == 1
    for target in ('{"result":19,"result":19}', '{"result":19.0}', '{"result":true}', '{"result":20}'):
        with pytest.raises(ValueError):
            validate_original_example({**a, "target": target})
