from __future__ import annotations


def load_eval_pack(pack_name: str) -> dict[str, object]:
    return {
        "pack_name": pack_name,
        "required_metrics": ["probe_score", "toxicity"],
        "description": "dry-run eval pack",
    }
