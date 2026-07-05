"""Prompt template loading for bounded LLM role calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Mapping


class PromptLoaderError(ValueError):
    """Raised when a prompt template cannot be loaded or rendered."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A prompt template loaded from the repository prompt directory."""

    name: str
    path: str
    text: str

    def render(self, values: Mapping[str, object] | None = None) -> str:
        """Render the template with deterministic ``str.format`` substitution."""

        payload = dict(values or {})
        required = {field for _, field, _, _ in Formatter().parse(self.text) if field}
        missing = sorted(field for field in required if field not in payload)
        if missing:
            raise PromptLoaderError(f"prompt '{self.name}' missing template values: {', '.join(missing)}")
        try:
            return self.text.format(**payload)
        except Exception as exc:  # pragma: no cover - defensive; missing handled above.
            raise PromptLoaderError(f"failed to render prompt '{self.name}': {exc}") from exc


@dataclass(frozen=True, slots=True)
class PromptLoader:
    """Load markdown prompt templates from a single prompt root."""

    prompt_dir: Path = Path("prompts")

    def load(self, name: str) -> PromptTemplate:
        safe_name = name.removesuffix(".md")
        if not safe_name or "/" in safe_name or "\\" in safe_name or safe_name in {".", ".."}:
            raise PromptLoaderError(f"invalid prompt name: {name!r}")
        path = self.prompt_dir / f"{safe_name}.md"
        if not path.exists() or not path.is_file():
            raise PromptLoaderError(f"missing prompt template: {path}")
        text = path.read_text(encoding="utf-8")
        return PromptTemplate(name=safe_name, path=str(path), text=text)

    def render(self, name: str, values: Mapping[str, object] | None = None) -> str:
        return self.load(name).render(values)
