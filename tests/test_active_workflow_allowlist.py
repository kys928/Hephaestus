"""Guard the active GitHub Actions surface after issue #151 R10 integration.

Only reviewed CPU integration/contract CI is allowed in the default tree.
Historical RunPod/S3/one-off actions live in Git ancestry, not this directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

ALLOWED = frozenset(
    {
        "autonomous-loop-integration.yml",
        "certified-role-model-stack.yml",
        "integration-151-foundation-no-gpu.yml",
        "integration-151-r10-runtime-cpu.yml",
        "production-autonomy-integration.yml",
    }
)

def test_only_reviewed_workflows_are_active() -> None:
    actual = {p.name for p in WORKFLOWS.iterdir() if p.is_file() and p.suffix in {".yml", ".yaml"}}
    assert actual == ALLOWED, (
        f"Active workflow change requires a reviewed allowlist update. "
        f"Unexpected: {sorted(actual - ALLOWED)}; missing: {sorted(ALLOWED - actual)}"
    )

def test_active_workflows_cannot_load_repository_secrets() -> None:
    # Historical campaign-specific workflows held provider credentials. New
    # privileged production actions require explicit review and separate gates.
    for name in sorted(ALLOWED):
        content = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "${{ secrets." not in content, f"Active workflow {name} references repository secrets"
        assert "RUNPOD_API_KEY" not in content, f"Active workflow {name} references a paid provider API key"
