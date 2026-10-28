"""Real full-source screening for prospective internal experiments.

Semantic checks are explicit: task-equivalence for supported generated tasks,
or a configured pinned embedding checker for external language. No lexical
heuristic is described as proof of semantic or historical independence.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import re
import unicodedata
from functools import lru_cache
from hephaestus.storage.verified import read_verified_json


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text):
    return set(re.findall(r"\w+", normalize(text)))


def similarity(a, b):
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x | y else 1.0


@lru_cache(maxsize=2)
def _verified_backend(raw):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(raw.decode())
    unk = tokenizer.token_to_id("<|unk|>")
    if unk is None:
        unk = tokenizer.token_to_id("<unk>")
    if unk is None and json.loads(raw)["model"].get("type") != "BPE":
        raise ValueError("unknown-free tokenizer must have explicit BPE representation")
    return tokenizer, unk


@dataclass(frozen=True)
class InternalSourceScanner:
    artifacts: object
    tokenizer_path: object
    tokenizer_sha256: str
    semantic_checker: object
    maximum_rows: int = 2000
    maximum_bytes: int = 8 * 1024 * 1024
    near_threshold: float = 0.90

    def token_ids(self, text):
        raw = self.tokenizer_path.read_bytes()
        if "sha256:" + sha256(raw).hexdigest() != self.tokenizer_sha256:
            raise ValueError("frozen tokenizer bytes changed")
        tokenizer, unk = _verified_backend(raw)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        return ids, sum(t == unk for t in ids) if unk is not None else 0

    def load_rows(self, source_file_hashes):
        rows, total_bytes = [], 0
        for ref in source_file_hashes.values():
            with self.artifacts.open(ref) as stream:
                raw = stream.read(self.maximum_bytes + 1)
            total_bytes += len(raw)
            if total_bytes > self.maximum_bytes or "sha256:" + sha256(raw).hexdigest() != ref:
                raise ValueError("source hash or byte budget mismatch")
            for line in raw.decode("utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
                    if len(rows) > self.maximum_rows:
                        raise ValueError("full-source scan row budget exceeded; partial scans cannot admit")
        if not rows:
            raise ValueError("empty source")
        return rows, total_bytes

    def scan(self, source_file_hashes, *, protocol_ref, processing_profile_sha256):
        rows, size = self.load_rows(source_file_hashes)
        protocol = read_verified_json(self.artifacts, protocol_ref)
        refs = []
        for field in ("evaluation_ref", "development_ref"):
            refs.extend(read_verified_json(self.artifacts, protocol[field])["examples"])
        if not refs:
            raise ValueError("prospective reference texts missing")
        malformed, pii, total, unknown, duplicates = 0, 0, 0, 0, 0
        exact, near, resolved, seen, prompts = 0, 0, 0, set(), []
        ref_prompts = [r["prompt"] for r in refs]
        reference_keys = {normalize(p) for p in ref_prompts}
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or
                    not isinstance(row.get("target"), str) or not row["prompt"].strip() or
                    not row["target"].strip()):
                malformed += 1
                continue
            prompt, text = row["prompt"], row["prompt"] + "\n" + row["target"]
            key = normalize(prompt)
            duplicates += key in seen
            seen.add(key)
            exact += key in reference_keys
            for reference in refs:
                if key not in reference_keys and similarity(prompt, reference["prompt"]) >= self.near_threshold:
                    resolver = getattr(self.semantic_checker, "distinct_task", None)
                    if callable(resolver) and resolver(row, reference):
                        resolved += 1
                    else:
                        near += 1
                        break
            pii += bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+?\d[ ()-]?){9,}", text))
            ids, unk = self.token_ids(text)
            total += len(ids)
            unknown += unk
            prompts.append(prompt)
        semantic = self.semantic_checker.check(rows, refs)
        if (not isinstance(semantic, dict) or type(semantic.get("overlap")) is not int or
                semantic["overlap"] < 0 or not semantic.get("method") or not semantic.get("limitations")):
            raise ValueError("complete semantic overlap implementation required")
        return {"schema": "internal-source-scan.v1", "coverage": "full_corpus",
            "source_file_hashes": source_file_hashes, "protocol_ref": protocol_ref,
            "processing_profile_sha256": processing_profile_sha256,
            "tokenizer_sha256": self.tokenizer_sha256, "rows": len(rows), "bytes": size,
            "malformed": malformed, "pii_flags": pii, "duplicates": max(duplicates, semantic.get("within_source_semantic_duplicates", 0)),
            "exact_overlap": exact, "near_overlap": near, "lexical_candidates_resolved_as_distinct_tasks": resolved, "semantic_overlap": semantic["overlap"],
            "semantic_method": semantic["method"], "semantic_evidence": semantic,
            "total_tokens": total, "unknown_tokens": unknown,
            "limitations": ["Internal prospective references only; historical/pretraining contamination unknown.",
                *semantic["limitations"]], "external_certification": False}


@dataclass
class PinnedEmbeddingOverlapChecker:
    """Actual pretrained encoder; caller pins model and tokenizer file hashes.

    Cosine flags are screening signals, not certified semantic isolation.
    """
    snapshot_dir: object
    expected_hashes: dict[str, str]
    threshold: float = 0.92

    def check(self, rows, references):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not 0 < self.threshold <= 1 or not {"model.safetensors", "tokenizer.json", "config.json"}.issubset(self.expected_hashes):
            raise ValueError("pinned embedding model, tokenizer and configuration required")
        for name, digest in self.expected_hashes.items():
            if name not in {"model.safetensors", "tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"}:
                raise ValueError("unexpected embedding artifact name")
            if "sha256:" + sha256((self.snapshot_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError("embedding identity changed")
        tok = AutoTokenizer.from_pretrained(self.snapshot_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(self.snapshot_dir, local_files_only=True, trust_remote_code=False).eval().to("cpu")
        def encode(texts):
            vectors = []
            for start in range(0, len(texts), 16):
                inputs = tok(texts[start:start+16], return_tensors="pt", padding=True, truncation=False)
                if inputs.input_ids.shape[1] > min(512, getattr(model.config, "max_position_embeddings", 512)):
                    raise ValueError("semantic scanner cannot truncate original text")
                with torch.inference_mode():
                    out = model(**inputs).last_hidden_state
                    mask = inputs.attention_mask.unsqueeze(-1)
                    vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                    vectors.append(torch.nn.functional.normalize(vec, dim=-1))
            return torch.cat(vectors)
        train = encode([r["prompt"] for r in rows])
        evaluation = encode([r["prompt"] for r in references])
        scores = train @ evaluation.T
        return {"overlap": int((scores.max(1).values >= self.threshold).sum().item()),
            "method": "pinned-pretrained-encoder-mean-pooling-cosine.v1",
            "model_hashes": self.expected_hashes, "threshold": self.threshold,
            "maximum_cosine": float(scores.max().item()),
            "limitations": ["Embedding cosine can miss paraphrases or flag legitimate related tasks."]}
