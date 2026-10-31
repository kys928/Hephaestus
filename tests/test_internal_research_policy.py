from dataclasses import replace
import pytest
from hephaestus.schemas.internal_research import InternalResearchPolicy
from hephaestus.state.prospective_evaluation import ProspectiveEvaluationStore
from hephaestus.policy.internal_research import InternalDatasetAdmissionPolicy, put_record
from hephaestus.policy.dataset_review import IndependentDatasetReviewPolicy
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def setup_policy(tmp_path):
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    state = SQLiteStateRepository(tmp_path / "state.sqlite")
    store = ProspectiveEvaluationStore(state, artifacts)
    material = put_record(artifacts, {"examples": [{"prompt": "new", "target": "prospective"}]})
    development = put_record(artifacts, {"examples": [{"prompt": "development", "target": "distinct"}]})
    protocol = store.freeze(experiment_id="experiment", evaluation_ref=material,
        development_ref=development, method="prospective comparison", limitations=["internal only"])
    policy = InternalResearchPolicy("policy", "user-authorized-research", "experiment", "workflow", "run",
        protocol, "sha256:" + "a" * 64, "sha256:" + "b" * 64, 4096, 100, 2, 120,
        ("CC0-1.0",), allowed_capabilities=("arithmetic",))
    ref = put_record(artifacts, policy.to_dict())
    admission = InternalDatasetAdmissionPolicy(artifacts, store, ref, frozenset({ref}))
    return artifacts, store, policy, admission


def test_internal_protocol_freezes_before_exposure_and_cannot_rewrite(tmp_path):
    artifacts, store, policy, admission = setup_policy(tmp_path)
    assert admission.policy().protocol_ref == policy.protocol_ref
    source = put_record(artifacts, {"source": "training"})
    store.record_exposure("experiment", policy.protocol_ref, source)
    with pytest.raises(PermissionError, match="already exists"):
        store.freeze(experiment_id="experiment", evaluation_ref=source, development_ref=source,
                     method="rewrite", limitations=["bad"])
    with pytest.raises(PermissionError, match="before exposure"):
        store.record_exposure("foreign", policy.protocol_ref, source)


def test_internal_policy_cannot_grant_itself_trust(tmp_path):
    _, _, _, admission = setup_policy(tmp_path)
    with pytest.raises(PermissionError, match="preauthorized"):
        replace(admission, trusted_policy_refs=frozenset()).policy()


@pytest.mark.parametrize("field,value", [("maximum_rows", True), ("maximum_bytes", 0),
    ("maximum_cpu_seconds", -1), ("maximum_training_steps", float("inf")),
    ("protocol_ref", "mutable"), ("allowed_capabilities", ()), ("authorized_by", "")])
def test_invalid_budgets_and_scope_fail_closed(tmp_path, field, value):
    _, _, policy, _ = setup_policy(tmp_path)
    with pytest.raises(ValueError):
        replace(policy, **{field: value}).validate()


def test_claim_selection_preserves_external_custody_default(tmp_path):
    artifacts, _, _, admission = setup_policy(tmp_path)
    external = IndependentDatasetReviewPolicy(artifacts, "tokenizer", "sha")
    assert external.for_claim("historical_external") is external
    assert external.for_claim("internal_prospective", internal_policy=admission) is admission
    with pytest.raises(PermissionError):
        external.for_claim("internal_prospective")
    with pytest.raises(PermissionError):
        external.for_claim("certified_without_custodian", internal_policy=admission)


def test_unknown_free_tokenizer_cannot_silently_drop_unrepresented_text(tmp_path):
    pytest.importorskip('tokenizers')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from hashlib import sha256
    from hephaestus.data.internal_screening import InternalSourceScanner
    tok = Tokenizer(models.BPE({'a':0}, []))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path/'incomplete-bpe.json'
    tok.save(str(path))
    scanner = InternalSourceScanner(None,path,'sha256:'+sha256(path.read_bytes()).hexdigest(),None)
    with pytest.raises(ValueError,match='byte-level BPE decoder'):
        scanner.token_ids('another symbol outside vocabulary')
