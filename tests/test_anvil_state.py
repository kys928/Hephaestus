from dataclasses import replace

import pytest

from hephaestus.schemas.anvil import Contribution, ContributionKind, ContributionOrigin
from hephaestus.state.anvil import AnvilState
from hephaestus.storage.sqlite import SQLiteStateRepository


def test_anvil_reconstructs_original_acceptance_from_existing_sqlite_repository(tmp_path):
    path = tmp_path / "anvil.sqlite"

    def bind():
        return AnvilState(SQLiteStateRepository(path), lambda _: True, lambda _, __: False)

    board = bind()
    empty = board.read_snapshot("study")
    origin = ContributionOrigin("study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
                                empty.frontier, "proposal-1", "org/model", "0" * 40, "sha256:" + "f" * 64)
    hypothesis = Contribution.from_model({"kind": "hypothesis", "text": "The wrapper is inconsistent."})
    first = board.append_contribution(origin, hypothesis)
    board = bind()
    assert board.append_contribution(origin, hypothesis) == first
    assert board.reconcile("study", origin.idempotency_key) == first
    assert board.reconcile("other", origin.idempotency_key) is None
    with pytest.raises(ValueError, match="conflicting"):
        board.append_contribution(origin, replace(hypothesis, text="Different claim"))
    response = Contribution.from_model({
        "kind": "challenge", "text": "There are counterexamples.",
        "links": [["challenges", first.contribution_id]],
    })
    reply_origin = replace(origin, idempotency_key="challenge-1", role="evaluator")
    with pytest.raises(ValueError, match="out-of-scope"):
        board.append_contribution(reply_origin, response)
    second = board.append_contribution(replace(reply_origin, context_frontier=first.frontier), response)
    assert board.replay("study", expected_frontier=second.frontier).query(
        kind=ContributionKind.CHALLENGE, linked_to=first.contribution_id,
    ) == (second,)
    assert board.read_snapshot("study", sequence=1).contributions == (first,)
    assert board.read_snapshot("study", sequence=0) == empty
    with pytest.raises(ValueError, match="retained frontier"):
        board.replay("study", expected_frontier=first.frontier)


def test_anvil_rejects_model_owned_authority_and_unbound_scope(tmp_path):
    state = SQLiteStateRepository(tmp_path / "anvil.sqlite")
    board = AnvilState(state, lambda _: True, lambda _, __: False)
    root = board.read_snapshot("study")
    origin = ContributionOrigin("study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
                                root.frontier, "proposal-1", "org/model", "0" * 40, "sha256:" + "f" * 64)
    for invalid in (
        {"kind": "hypothesis", "text": "claim", "authorization": True},
        {"kind": "experiment_result", "text": "test passed"},
        {"kind": "revision", "text": "revision without parent"},
    ):
        with pytest.raises(ValueError):
            Contribution.from_model(invalid)
    hypothesis = Contribution.from_model({"kind": "hypothesis", "text": "claim"})
    first = board.append_contribution(origin, hypothesis)
    request = Contribution.from_model({
        "kind": "capability_request", "text": "Measure coverage", "capability": "diagnose_coverage",
    })
    with pytest.raises(ValueError, match="current investigation frontier"):
        board.append_contribution(replace(origin, idempotency_key="request"), request)
    followup = replace(origin, idempotency_key="followup", context_frontier=first.frontier)
    with pytest.raises(PermissionError, match="scoped verification"):
        board.append_contribution(followup, replace(hypothesis, evidence_refs=("sha256:" + "b" * 64,)))
    challenge = Contribution.from_model({
        "kind": "challenge", "text": "unbound reference",
        "links": [["challenges", first.contribution_id]],
    })
    with pytest.raises(ValueError, match="out-of-scope"):
        board.append_contribution(replace(followup, lineage_id="other"), challenge)
    board.verify_origin = lambda _: False
    with pytest.raises(PermissionError, match="receipt"):
        board.append_contribution(followup, hypothesis)


def test_checkpoint_is_hash_bound_and_cannot_roll_back_accepted_state(tmp_path):
    from hephaestus.storage.filesystem import FileSystemArtifactStore

    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    source = AnvilState(SQLiteStateRepository(tmp_path / "source.sqlite"), lambda _: True, lambda *_: False)
    destination = AnvilState(SQLiteStateRepository(tmp_path / "destination.sqlite"), lambda _: True, lambda *_: False)
    origin = ContributionOrigin("study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
                                source.read_snapshot("study").frontier, "first", "org/model", "0" * 40, "sha256:" + "f" * 64)
    payload = Contribution.from_model({"kind": "hypothesis", "text": "claim"})
    first = source.append_contribution(origin, payload)
    old = source.checkpoint("study", artifacts)
    latest = source.append_contribution(replace(origin, idempotency_key="second"),
                                       replace(payload, text="second claim"))
    sealed = source.checkpoint("study", artifacts)
    with pytest.raises(ValueError, match="retained scope"):
        destination.restore("study", artifacts, old, expected_frontier=latest.frontier)
    restored = destination.restore("study", artifacts, sealed, expected_frontier=latest.frontier)
    assert restored == source.read_snapshot("study")
    assert destination.restore("study", artifacts, sealed, expected_frontier=latest.frontier) == restored
    with pytest.raises(ValueError, match="roll back"):
        destination.restore("study", artifacts, old, expected_frontier=first.frontier)
    assert destination.replay("study", expected_frontier=latest.frontier) == restored


def test_uncertain_acknowledgement_reconciles_identity_without_second_append(tmp_path, monkeypatch):
    from hephaestus.storage.sqlite import StateCommitUncertain

    path = tmp_path / "anvil.sqlite"
    repository = SQLiteStateRepository(path)
    board = AnvilState(repository, lambda _: True, lambda *_: False)
    origin = ContributionOrigin(
        "study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
        board.read_snapshot("study").frontier, "first", "org/model", "0" * 40, "sha256:" + "f" * 64,
    )
    payload = Contribution.from_model({"kind": "hypothesis", "text": "claim"})
    committed = repository.append

    def lose_ack(*args, **kwargs):
        committed(*args, **kwargs)
        raise StateCommitUncertain(path)

    monkeypatch.setattr(repository, "append", lose_ack)
    with pytest.raises(StateCommitUncertain):
        board.append_contribution(origin, payload)
    restarted = AnvilState(SQLiteStateRepository(path), lambda _: True, lambda *_: False)
    assert restarted.append_contribution(origin, payload) == restarted.reconcile("study", "first")
    assert len(restarted.read_snapshot("study").contributions) == 1


def test_triads_retained_v1_outputs_preserve_negative_control_semantics():
    from pathlib import Path

    from hephaestus.llm.anvil_legacy import project_legacy_answer
    from hephaestus.llm.role_contract import RoleMasteryContract

    # Exact output tails recovered by independent observation run 45376969755.
    diagnosis = (
        '{"decision":"data_format_or_wrapper","action":"change_preprocessing",'
        '"primary_variable":"preprocessing_policy","confidence":0.96,'
        '"evidence_refs":["triad-case-1"],"uncertainties":[],'
        '"rationale":"data_format_wrapper: apply only the material evidence and the bounded diagnosis contract."}'
    )
    planner = (
        '{"decision":"repair_evaluation","action":"request_recheck",'
        '"primary_variable":"evaluation_protocol","confidence":0.96,'
        '"evidence_refs":["triad-case-1"],"uncertainties":[],'
        '"rationale":"evaluation_first: apply only the material evidence and the bounded planner contract."}'
    )
    contract = RoleMasteryContract.load(
        Path(__file__).resolve().parents[1] / "configs/models/hephaestus_role_mastery_runtime_v1.json",
    )
    observed_diagnosis = project_legacy_answer(
        contract, role="diagnosis", raw=diagnosis, allowed_refs={"triad-case-1"},
        receipt_ref="sha256:9f231c528d1e13431637b873b96f4985e7fd919f564576affe0bc8b132b850d4",
    )
    observed_planner = project_legacy_answer(
        contract, role="planner", raw=planner, allowed_refs={"triad-case-1"},
        receipt_ref="sha256:a9f86985dbef280461324aa6b1a4c650ad52f2d434451b2a486601cba2818f36",
    )
    assert observed_diagnosis.answer.action == "change_preprocessing"
    answer = observed_planner.answer
    triple = (answer.decision, answer.action, answer.primary_variable)
    assert triple == ("repair_evaluation", "request_recheck", "evaluation_protocol")
    assert triple != ("collect_more_evidence", "request_recheck", "diagnostic_measurement")
