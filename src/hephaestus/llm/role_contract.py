"""Frozen role-mastery prompt and response contract.

Inference mirrors the original role-mastery v1 instruction style and exact
vocabularies. Responses are *proposals*, never permissions or machine facts.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hephaestus.providers.models.role_stack import REQUIRED_ROLES


# Base identities independently recovered from the frozen training config.
BASE_IDENTITIES = {'controller': ('Qwen/Qwen3-4B-Instruct-2507', 'cdbee75f17c01a7cc42f958dc650907174af0554'), 'diagnosis': ('mistralai/Ministral-3-3B-Reasoning-2512', '4a36357c811bf511a7b625d132e12f22408aac91'), 'planner': ('mistralai/Ministral-3-8B-Reasoning-2512', '81eaece1948f3875421d9a45bc55487d10e2d894'), 'evaluator': ('ibm-granite/granite-3.3-8b-instruct', '51dd4bc2ade4059a6bd87649d68aa11e4fb2529b'), 'judge': ('microsoft/Phi-4-mini-instruct', 'cfbefacb99257ffa30c83adab238a50856ac3083')}


KEYS = ("decision", "action", "primary_variable", "confidence",
        "evidence_refs", "uncertainties", "rationale")


@dataclass(frozen=True, slots=True)
class RoleEvidence:
    ref: str
    fact: str


@dataclass(frozen=True, slots=True)
class RoleTask:
    role: str
    situation: str
    evidence: tuple[RoleEvidence, ...]
    run_id: str
    lineage_id: str
    stage_name: str


@dataclass(frozen=True, slots=True)
class RoleAnswer:
    decision: str
    action: str
    primary_variable: str
    confidence: float
    evidence_refs: tuple[str, ...]
    uncertainties: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {"decision": self.decision, "action": self.action,
                "primary_variable": self.primary_variable, "confidence": self.confidence,
                "evidence_refs": list(self.evidence_refs),
                "uncertainties": list(self.uncertainties), "rationale": self.rationale}


class RoleMasteryContract:
    def __init__(self, payload: Mapping[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        # Independently compared with VOCAB/ROLE_RULES and candidate profiles at
        # frozen training commit c831bbfc0e2054d54fccd1ef374fd84722612c79.
        if hashlib.sha256(encoded.encode()).hexdigest() != (
            "29bdc8a987403fd2871b4d4b389ed788edb278d833c93df13581a247736ba163"
        ):
            raise ValueError("role runtime contract integrity digest drift")
        self._encoded = encoded
        if self.payload.get("contract_id") != "hephaestus-role-mastery-runtime.v1":
            raise ValueError("unsupported role mastery contract")
        if self.payload.get("original_training_pack_sha256") != (
            "6f4bb3b8c8312a4189136ea961e8fa32629c02fa5459e18fa960670badb203b3"
        ):
            raise ValueError("role mastery training pack provenance mismatch")
        if self.payload.get("required_exact_keys") != list(KEYS):
            raise ValueError("role mastery output schema drift")
        roles = self.payload.get("roles")
        if not isinstance(roles, dict) or set(roles) != set(REQUIRED_ROLES):
            raise ValueError("role mastery contract must specify exactly five roles")
        for role, config in roles.items():
            if not isinstance(config, dict) or not isinstance(config.get("rule"), str):
                raise ValueError(f"invalid role rule: {role}")
            vocabulary = config.get("vocabulary")
            if not isinstance(vocabulary, dict):
                raise ValueError(f"invalid role vocabulary: {role}")
            for field in ("decision", "action", "primary_variable"):
                values = vocabulary.get(field)
                if not isinstance(values, list) or not values or not all(
                    isinstance(x, str) and x for x in values
                ) or len(set(values)) != len(values):
                    raise ValueError(f"invalid {role}.{field} vocabulary")
            if config.get("load_kind") not in ("causal_lm", "mistral3"):
                raise ValueError(f"unrecognized role model kind: {role}")
            if not isinstance(config.get("decoding"), dict):
                raise ValueError(f"missing frozen decoding: {role}")

    @property
    def payload(self) -> dict[str, object]:
        return json.loads(self._encoded)

    @classmethod
    def load(cls, path: str | Path) -> "RoleMasteryContract":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def profile(self, role: str) -> dict[str, object]:
        if role not in REQUIRED_ROLES:
            raise ValueError(f"unknown Hephaestus role: {role}")
        return dict(self.payload["roles"][role])

    def validate_model(self, spec) -> None:
        if BASE_IDENTITIES.get(spec.role) != (spec.model_id, spec.revision):
            raise ValueError("certified base model identity/revision drift from frozen contract")

    def prompt(self, task: RoleTask) -> str:
        cfg = self.profile(task.role)
        if not task.situation.strip() or len(task.situation) > 12_000:
            raise ValueError("role situation must be nonempty and bounded")
        if not task.run_id or not task.lineage_id or not task.stage_name:
            raise ValueError("role task is missing workflow identity")
        if not task.evidence or len(task.evidence) > 100:
            raise ValueError("role task requires bounded evidence")
        refs = [e.ref for e in task.evidence]
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:/._#?=&%+-]{0,255}", x)
               for x in refs) or len(set(refs)) != len(refs):
            raise ValueError("evidence identifiers must be safe, nonempty and unique")
        if any(not x.fact or len(x.fact) > 3000 for x in task.evidence):
            raise ValueError("evidence facts must be nonempty and bounded")
        # Preserve the original training bullet format, without allowing
        # untrusted newlines to spoof another evidence item or control section.
        evidence = "\n".join(
            f"- {e.ref}: {' '.join(e.fact.splitlines())}" for e in task.evidence
        )
        v = cfg["vocabulary"]
        return (
            f"You are the Hephaestus {task.role.upper()} specialist.\n\n"
            f"ROLE CONTRACT:\n{cfg['rule']}\n\n"
            f"SITUATION:\n{task.situation}\n\nEVIDENCE:\n{evidence}\n\n"
            "Return exactly one JSON object and nothing else. No markdown.\n"
            f"The object must contain exactly these keys: {', '.join(KEYS)}.\n"
            "Use only the role vocabulary:\n"
            f"- decision: {', '.join(v['decision'])}\n"
            f"- action: {', '.join(v['action'])}\n"
            f"- primary_variable: {', '.join(v['primary_variable'])}\n"
            "- confidence: number in [0,1]\n"
            "- evidence_refs: cite only material evidence from this case\n"
            "- uncertainties: short JSON array\n"
            "- rationale: one concise evidence-grounded sentence\n\n"
            "Do not perform another role's job. Do not invent evidence. "
            "Do not reveal hidden reasoning."
        )

    def parse(self, role: str, raw: str, allowed_refs: set[str]) -> RoleAnswer:
        cfg = self.profile(role)
        # The two Mistral3 reasoning models have documented end-of-thinking
        # projections in their original frozen training configuration.
        projected = raw.strip()
        for stop in cfg.get("reasoning_projection", []):
            if stop in projected:
                projected = projected.rsplit(stop, 1)[-1].strip()
        try:
            data = json.loads(projected)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("model did not return one complete JSON object") from exc
        if not isinstance(data, dict) or set(data) != set(KEYS):
            raise ValueError("role response must have exactly the frozen seven keys")
        vocab = cfg["vocabulary"]
        for field in ("decision", "action", "primary_variable"):
            if not isinstance(data[field], str) or data[field] not in vocab[field]:
                raise ValueError(f"role response has invalid {field}")
        conf = data["confidence"]
        if type(conf) not in (float, int) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("role response confidence is invalid")
        refs = data["evidence_refs"]
        if not isinstance(refs, list) or not refs or not all(
            isinstance(x, str) and x in allowed_refs for x in refs
        ) or len(set(refs)) != len(refs):
            raise ValueError("role response cites missing, duplicate or empty evidence")
        un = data["uncertainties"]
        if not isinstance(un, list) or len(un) > 30 or not all(
            isinstance(x, str) and len(x) <= 300 for x in un
        ):
            raise ValueError("role response uncertainties are malformed")
        rationale = data["rationale"]
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2000:
            raise ValueError("role response rationale is malformed")
        return RoleAnswer(data["decision"], data["action"], data["primary_variable"],
                          float(conf), tuple(refs), tuple(un), rationale)
