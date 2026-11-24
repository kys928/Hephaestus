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


def _append_in_child(path, origin, payload, start, *, exit_boundary=None):
    import os
    import time

    from hephaestus.state.single_writer import WriterBusyError

    repository = SQLiteStateRepository(path, initialize=False)
    board = AnvilState(repository, lambda _: True, lambda *_: False)
    if exit_boundary == "before_commit":
        def append_then_exit(collection, row, **options):
            with repository.transaction() as transaction:
                transaction.append(collection, row, **options)
                os._exit(41)

        repository.append = append_then_exit
    elif exit_boundary == "after_commit":
        committed = repository.append

        def append_then_exit(*args, **kwargs):
            committed(*args, **kwargs)
            os._exit(42)

        repository.append = append_then_exit
    if not start.wait(10):
        raise TimeoutError("fixture start was not signalled")
    for _ in range(200):
        try:
            board.append_contribution(origin, payload)
            return
        except WriterBusyError:
            time.sleep(0.01)
    raise TimeoutError("writer fence never became available")


def test_multiple_processes_and_interrupted_commits_preserve_accepted_frontier(tmp_path):
    import multiprocessing

    path = tmp_path / "anvil.sqlite"
    board = AnvilState(SQLiteStateRepository(path), lambda _: True, lambda *_: False)
    empty = board.read_snapshot("study")
    origin = ContributionOrigin(
        "study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
        empty.frontier, "first", "org/model", "0" * 40, "sha256:" + "f" * 64,
    )
    payload = Contribution.from_model({"kind": "hypothesis", "text": "claim"})
    context = multiprocessing.get_context("fork")
    start = context.Event()
    workers = [
        context.Process(target=_append_in_child, args=(path, replace(origin, idempotency_key=key), payload, start))
        for key in ("first", "second", "third", "first")
    ]
    try:
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        committed = board.read_snapshot("study")
        assert len(committed.contributions) == 3

        for mode, code in (("before_commit", 41), ("after_commit", 42)):
            worker = context.Process(
                target=_append_in_child,
                args=(path, replace(origin, idempotency_key=mode), payload, start),
                kwargs={"exit_boundary": mode},
            )
            workers.append(worker)
            worker.start()
            worker.join(10)
            assert worker.exitcode == code

        restarted = AnvilState(SQLiteStateRepository(path), lambda _: True, lambda *_: False)
        assert restarted.reconcile("study", "before_commit") is None
        receipt = restarted.reconcile("study", "after_commit")
        assert receipt is not None
        assert restarted.append_contribution(replace(origin, idempotency_key="after_commit"), payload) == receipt
        assert restarted.replay("study", expected_frontier=receipt.frontier).contributions[:3] == committed.contributions
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()


def test_cold_admission_rebinds_fixture_receipt_to_exact_context(tmp_path):
    from hashlib import sha256

    from hephaestus.control.anvil_context import AnvilAdmission
    from hephaestus.production.original_cuda_identity import registry_digest, verified_original_roles
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    from hephaestus.storage.verified import canonical_bytes

    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")

    def publish(document):
        return artifacts.put_bytes(canonical_bytes(document), media_type="application/json").artifact_ref

    registry = registry_digest()
    stack, contract = verified_original_roles(registry)
    role = "planner"
    model = stack.resolve(role)
    scope = {"investigation": "study", "run_id": "run", "lineage_id": "lineage"}
    objective = publish(dict(scope, schema="anvil.material.v1", kind="objective", text="Compare two hypotheses."))
    evidence = publish(dict(scope, schema="anvil.material.v1", kind="evidence", text="Fixture measurement."))
    materials = frozenset((objective, evidence))
    repository = SQLiteStateRepository(tmp_path / "board.sqlite")
    board = AnvilState(repository, lambda _: True,
                       lambda _, ref: ref in materials)
    admission = AnvilAdmission(board, artifacts, registry, materials, frozenset(), repository)
    origin = ContributionOrigin(**scope, role=role, inference_ref=objective,
                                context_frontier=board.read_snapshot("study").frontier,
                                idempotency_key="first", model_id=model.model_id, model_revision=model.revision,
                                adapter_ref="sha256:" + model.adapter.sha256)
    with pytest.raises(PermissionError, match="independent admission"):
        admission.prepare(origin, 0, (objective, evidence))
    for ref in materials:
        admission.admit_material(origin, ref)
        admission.admit_material(origin, ref)
    admission = replace(admission, trusted_material=frozenset())
    task = admission.prepare(origin, 0, (objective, evidence))
    assert task == admission.prepare(origin, 0, (evidence, objective))
    with pytest.raises(PermissionError, match="independent admission"):
        admission.prepare(origin, 0, (objective, "sha256:" + "f" * 64))
    vocabulary = contract.profile(role)["vocabulary"]
    answer = {key: vocabulary[key][0] for key in ("decision", "action", "primary_variable")}
    answer.update(confidence=0.6, uncertainties=["One fixture"], evidence_refs=[evidence],
                  rationale="The evidence favors the first hypothesis.")
    raw, prompt = canonical_bytes(answer).decode(), contract.prompt(task)
    receipt = publish(dict(
        schema="role-inference.v1", status="accepted", role=role, run_id=task.run_id,
        lineage_id=task.lineage_id, stage_name=task.stage_name, raw_output=raw,
        output_sha256=sha256(raw.encode()).hexdigest(), prompt=prompt,
        prompt_sha256=sha256(prompt.encode()).hexdigest(), answer=answer,
        model_id=model.model_id, base_revision=model.revision, adapter_sha256=model.adapter.sha256,
        adapter_source=model.adapter.s3_key, tokenizer_id=model.model_id, tokenizer_revision=model.revision,
        evidence_refs=[item.ref for item in task.evidence],
    ))
    origin = replace(origin, inference_ref=receipt)
    with pytest.raises(PermissionError, match="independent admission"):
        admission.accept(origin, 0, (objective, evidence))
    admission = replace(admission, trusted_receipts=frozenset((receipt,)))
    with pytest.raises(PermissionError, match="exact context"):
        admission.accept(origin, 0, (objective,))
    admitted = admission.accept(origin, 0, (objective, evidence))
    assert admitted.payload.kind == ContributionKind.MODEL_CLAIM
    assert admitted.payload.v1_position.action == answer["action"]
    next_task = admission.prepare(replace(origin, context_frontier=admitted.frontier), 1, (objective, evidence))
    assert '"primary_variable"' in next_task.situation and '"model_claim"' in next_task.situation
    assert admitted.payload.confidence == 0.6
    assert admitted.payload.uncertainties == ("One fixture",)
    assert board.replay("study").contributions == (admitted,)
    assert admission.accept(origin, 0, (evidence, objective)) == admitted
    with pytest.raises(PermissionError, match="frontier"):
        admission.accept(origin, 1, (objective, evidence))


def test_fixture_specialists_record_competing_positions_and_revision(tmp_path):
    board = AnvilState(SQLiteStateRepository(tmp_path / "collaboration.sqlite"),
                       lambda _: True, lambda *_: False)
    origin = ContributionOrigin("study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
                                board.read_snapshot("study").frontier, "diagnosis-1",
                                "org/model", "0" * 40, "sha256:" + "f" * 64)
    first = board.append_contribution(origin, Contribution.from_model({
        "kind": "hypothesis", "text": "Coverage is incomplete.", "confidence": 0.7,
    }))
    second = board.append_contribution(replace(origin, role="planner", idempotency_key="planner-1"),
                                      Contribution.from_model({
        "kind": "hypothesis", "text": "The index is stale.", "confidence": 0.6,
    }))
    challenged = board.append_contribution(replace(origin, role="evaluator", idempotency_key="eval-1",
                                                   context_frontier=second.frontier),
                                           Contribution.from_model({
        "kind": "challenge", "text": "The coverage hypothesis also predicts the missing entries.",
        "links": [["challenges", second.contribution_id]],
    }))
    revised = board.append_contribution(replace(origin, role="planner", idempotency_key="planner-2",
                                                context_frontier=challenged.frontier),
                                        Contribution.from_model({
        "kind": "revision", "text": "Both hypotheses remain viable pending an independent measurement.",
        "links": [["revises", second.contribution_id]], "uncertainties": ["Index freshness unmeasured."],
    }))
    request = board.append_contribution(replace(origin, role="planner", idempotency_key="planner-3",
                                                context_frontier=revised.frontier),
                                        Contribution.from_model({
        "kind": "capability_request", "capability": "diagnose_coverage", "text": "Measure coverage.",
    }))
    assert board.replay("study", expected_frontier=request.frontier).contributions == (
        first, second, challenged, revised, request,
    )
