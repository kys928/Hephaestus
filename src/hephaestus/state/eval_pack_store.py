from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.schemas.eval_pack import EvalPack
from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class EvalPackStore:
    root: Path

    def register(self, pack: EvalPack | dict[str, object]) -> EvalPack:
        normalized = self._normalize(pack)
        JsonStore(self.root, "eval_packs.jsonl").append(normalized.to_dict())
        return normalized

    def append(self, pack: EvalPack | dict[str, object]) -> EvalPack:
        return self.register(pack)

    def get(self, eval_pack_id: str) -> EvalPack | None:
        rows = JsonStore(self.root, "eval_packs.jsonl").all()
        for row in reversed(rows):
            normalized = EvalPack.normalize(dict(row))
            if normalized.eval_pack_id == eval_pack_id:
                return normalized
        return None

    def list_for_stage(self, stage_name: str) -> list[EvalPack]:
        rows = JsonStore(self.root, "eval_packs.jsonl").all()
        items = [EvalPack.normalize(dict(row)) for row in rows]
        return [item for item in items if item.stage_name == stage_name]

    def _normalize(self, pack: EvalPack | dict[str, object]) -> EvalPack:
        if isinstance(pack, EvalPack):
            normalized = pack
        else:
            normalized = EvalPack.normalize(dict(pack))
        if not normalized.frozen:
            normalized.frozen = True
            normalized.warnings.append("persisted_eval_pack_forced_frozen")
        if not normalized.mutation_policy:
            normalized.mutation_policy = "immutable_without_approval"
        return normalized
