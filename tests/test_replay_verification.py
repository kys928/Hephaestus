from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.cli.verify_replay import main as verify_replay_main
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.replay_verification import verify_run_replay
from hephaestus.state._json_store import JsonStore
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.run_store import RunStore


def _run_dry(tmp_path: Path, run_id: str = "replay-run") -> None:
    orch = build_orchestrator(state_root=tmp_path, run_id=run_id, backend=DryRunBackend())
    orch.run(run_id)


def _state_snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_dry_run_replay_verifies_as_reproducible_or_partial(tmp_path: Path) -> None:
    run_id = "replay-dry"
    _run_dry(tmp_path, run_id)

    report = verify_run_replay(tmp_path, run_id)

    assert report.status in {"reproducible", "partial"}
    assert report.run_id == run_id
    assert report.lineage_id
    assert report.manifest_id
    assert report.eval_report_id == f"eval-{run_id}"
    assert report.decision_id == f"dec-{run_id}-exit"
    assert report.checkpoint_ref
    assert "run_record" not in report.missing_evidence
    if report.status == "reproducible":
        assert report.content_hash_available is True or report.checkpoint_ref is None
    else:
        assert "checkpoint_content_hash_unavailable" in report.warnings


def test_missing_run_returns_missing(tmp_path: Path) -> None:
    report = verify_run_replay(tmp_path, "does-not-exist")

    assert report.status == "missing"
    assert report.missing_evidence == ["run_record"]


def test_missing_eval_report_or_decision_returns_insufficient(tmp_path: Path) -> None:
    run_id = "replay-missing-eval"
    _run_dry(tmp_path, run_id)
    (tmp_path / "reports.jsonl").write_text("", encoding="utf-8")

    missing_eval = verify_run_replay(tmp_path, run_id)

    assert missing_eval.status == "insufficient"
    assert "eval_report" in missing_eval.missing_evidence

    run_id_2 = "replay-missing-decision"
    _run_dry(tmp_path, run_id_2)
    rows = [row for row in DecisionStore(tmp_path).all() if row.get("decision_id") != f"dec-{run_id_2}-exit"]
    with (tmp_path / "decision_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    missing_decision = verify_run_replay(tmp_path, run_id_2)

    assert missing_decision.status == "insufficient"
    assert "judge_exit_decision" in missing_decision.missing_evidence


def test_missing_content_hash_when_required_returns_insufficient(tmp_path: Path) -> None:
    run_id = "replay-required-hash"
    _run_dry(tmp_path, run_id)
    run = RunStore(tmp_path).get(run_id)
    assert run is not None
    replay_metadata = dict(run.get("replay_metadata") or {})
    replay_metadata["requires_content_hash_match"] = True
    replay_metadata["content_hash_available"] = False
    replay_metadata.pop("checkpoint_content_hash", None)
    run["replay_metadata"] = replay_metadata
    JsonStore(tmp_path, "run_records.jsonl").append(run)

    report = verify_run_replay(tmp_path, run_id)

    assert report.status == "insufficient"
    assert "checkpoint_content_hash" in report.missing_evidence
    assert report.requires_content_hash_match is True
    assert report.content_hash_available is False


def test_cli_json_output_has_stable_top_level_keys(tmp_path: Path) -> None:
    run_id = "replay-cli-json"
    _run_dry(tmp_path, run_id)

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = verify_replay_main(["--state-root", str(tmp_path), "--run-id", run_id, "--format", "json"])

    assert code == 0
    payload = json.loads(buf.getvalue())
    assert list(payload.keys()) == sorted(
        [
            "run_id",
            "lineage_id",
            "status",
            "checked_at",
            "evidence_refs",
            "missing_evidence",
            "warnings",
            "replay_scope",
            "checkpoint_ref",
            "checkpoint_content_hash",
            "content_hash_available",
            "requires_content_hash_match",
            "manifest_id",
            "eval_report_id",
            "decision_id",
            "confidence_ceiling",
            "summary",
        ]
    )


def test_cli_text_output_contains_status_and_missing_evidence(tmp_path: Path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = verify_replay_main(["--state-root", str(tmp_path), "--run-id", "missing-run", "--format", "text"])

    assert code == 0
    output = buf.getvalue()
    assert "status: missing" in output
    assert "missing_evidence:" in output
    assert "run_record" in output


def test_verification_does_not_modify_state_files(tmp_path: Path) -> None:
    run_id = "replay-read-only"
    _run_dry(tmp_path, run_id)
    before = _state_snapshot(tmp_path)

    verify_run_replay(tmp_path, run_id)
    buf = io.StringIO()
    with redirect_stdout(buf):
        verify_replay_main(["--state-root", str(tmp_path), "--run-id", run_id, "--format", "json"])

    assert _state_snapshot(tmp_path) == before
