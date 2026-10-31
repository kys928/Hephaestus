"""Documentation contract: current guides and old stable references remain reachable.

This test never connects to external providers or modifies scientific evidence.
"""

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    "docs/README.md",
    "docs/architecture/README.md",
    "docs/architecture/control-and-governance.md",
    "docs/architecture/data-and-state.md",
    "docs/operations/README.md",
    "docs/operations/developer-quickstart.md",
    "docs/operations/run-safety.md",
    "docs/operations/release-checklist.md",
    "docs/scientific-reports/README.md",
    "docs/scientific-reports/r10-internal-cuda45.md",
    "docs/scientific-reports/scientific-claims.md",
)
STABLE = (
    "docs/architecture.md",
    "docs/control_spine.md",
    "docs/state_layout.md",
    "docs/runtime_monitoring.md",
    "docs/memory_model.md",
    "docs/scoring_and_eval.md",
    "docs/stage_policy.md",
    "docs/backend_contracts.md",
)
ROOT_ENTRYPOINTS = ("README.md", "PROJECT_MISSION.md", "REPO_STRUCTURE_PLAN.md", "AGENTS.md")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+[^)]+)?\)")


def test_canonical_hierarchy_and_replaced_todo_stubs() -> None:
    for rel in CANONICAL + STABLE:
        path = ROOT / rel
        assert path.is_file(), f"Missing documentation: {rel}"
        assert path.read_text(encoding="utf-8").strip()
    for rel in STABLE:
        assert "TODO: document concrete decisions" not in (ROOT / rel).read_text(encoding="utf-8")


def test_current_documentation_links_resolve_without_network() -> None:
    for rel in ROOT_ENTRYPOINTS + CANONICAL + STABLE:
        path = ROOT / rel
        markdown = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(markdown):
            url = match.group(1)
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or url.startswith(("#", "//")):
                continue
            target = unquote(parts.path)
            if not target:
                continue
            # pathlib resolves ../ relative to the source file. Historical
            # evidence stays in its original path, so links must not be faked.
            candidate = path.parent / target
            assert candidate.exists(), f"{rel}: broken relative link {url}"
