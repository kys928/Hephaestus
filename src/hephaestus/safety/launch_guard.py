from __future__ import annotations

from hephaestus.schemas.safety_guard import SafetyGuardResult


def evaluate_launch_guard(launch_config: dict[str, object], *, fixture_mode: bool = False) -> SafetyGuardResult:
    reasons: list[str] = []
    dry_run = launch_config.get("dry_run") is True
    if not dry_run and fixture_mode is not True:
        reasons.append("non_dry_run_launch_requires_approval")
    return SafetyGuardResult(
        guard_id="launch_guard",
        passed=not reasons,
        severity="error" if reasons else "info",
        reasons=reasons,
        metadata={"dry_run": dry_run, "legacy_fixture_mode": fixture_mode is True,
                  "scope": "legacy_demo_or_fixed_test_fixture_only"},
    )


def validate_legacy_fixture_job(prepared_job) -> None:
    """The opt-in legacy fixture route cannot execute a configurable trainer."""
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[3]
    spec = prepared_job.execution_spec
    if spec.get("runner") == "subprocess":
        command = spec.get("command", [])
        if (not isinstance(command, list) or len(command) < 2
                or Path(str(command[0])).resolve() != Path(sys.executable).resolve()
                or Path(str(command[1])).resolve() != root / "tests/fixtures/fake_trainer.py"):
            raise PermissionError("legacy fixture route cannot execute a configurable trainer")
    elif spec.get("runner") == "ardor_local_process":
        script = spec.get("job_spec", {}).get("runner_script", "")
        if Path(str(script)).resolve() != root / "tests/fixtures/fake_ardor_runner.py":
            raise PermissionError("legacy fixture route cannot execute a configurable Ardor runner")
    else:
        raise PermissionError("legacy fixture route requires a fixed local test fixture")
