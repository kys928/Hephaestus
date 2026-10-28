"""Original verifiable data with canonical-key split isolation and frozen token IDs."""
from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import random
from hephaestus.schemas.discovery_contract import DatasetCandidate
from hephaestus.schemas.internal_research import InternalDatasetReview
from hephaestus.policy.internal_research import put_record
from hephaestus.policy.dataset_review import candidate_sha256
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from .generated_validation import validate_original_example, OriginalTaskSemanticChecker

FORMAT = "original-task-token-ids.v1"


def generator_code_sha256():
    return "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()


def validator_code_sha256():
    from . import generated_validation
    return "sha256:" + sha256(Path(generated_validation.__file__).read_bytes()).hexdigest()


def _example(rng, family, difficulty):
    if family == "arithmetic":
        a, b = rng.randrange(difficulty), rng.randrange(difficulty)
        op = rng.choice(["plus", "minus", "times"])
        result = a + b if op == "plus" else a - b if op == "minus" else a * b
        prompt, target = f"Compute {a} {op} {b}. Return JSON with key result.", {"result": result}
    elif family == "json":
        values = [rng.randrange(difficulty) for _ in range(rng.randrange(2, 6))]
        prompt = f"For integers {json.dumps(values)}, return JSON with keys count and sum."
        target = {"sum": sum(values), "count": len(values)}
    elif family == "symbolic":
        text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randrange(3, 9)))
        prompt, target = f"Reverse the lowercase string {text}. Return JSON with key text.", {"text": text[::-1]}
    else:
        edges = sorted({(i, i+1) for i in range(4)} | {(rng.randrange(5), rng.randrange(5)) for _ in range(6)})
        queue, visited, answer = deque([(0, 0)]), {0}, None
        while queue:
            node, distance = queue.popleft()
            if node == 4:
                answer = distance
                break
            for a, b in edges:
                if a == node and b not in visited:
                    visited.add(b)
                    queue.append((b, distance + 1))
        prompt = f"Find shortest directed path length from 0 to 4 in edges {json.dumps(edges)}. Return JSON with key distance."
        target = {"distance": answer}
    return {"prompt": prompt, "target": json.dumps(target, sort_keys=True, separators=(",", ":")), "family": family}


def generate_original_splits(spec):
    spec.validate()
    rng = random.Random(spec.seed)
    wanted = {"train": spec.train_per_family, "development": spec.development_per_family,
              "evaluation": spec.evaluation_per_family}
    splits = {name: [] for name in wanted}
    seen, rejected = set(), Counter()
    for family in spec.families:
        counts = Counter()
        for _ in range(sum(wanted.values()) * 1000):
            row = _example(rng, family, spec.difficulty)
            key = validate_original_example(row)
            if key in seen:
                rejected["canonical_duplicates"] += 1
                continue
            # Global partition is independent of seed and generation order.
            # Any seed's training examples are excluded from every seed's eval partition.
            bucket = int(sha256(canonical_bytes(key)).hexdigest(), 16) % 10
            split = "development" if bucket == 8 else "evaluation" if bucket == 9 else "train"
            if counts[split] >= wanted[split]:
                continue
            seen.add(key)
            counts[split] += 1
            row["sample_id"] = "sha256:" + sha256(canonical_bytes(key)).hexdigest()
            splits[split].append(row)
            if all(counts[name] == count for name, count in wanted.items()):
                break
        else:
            raise ValueError("original generation budget exhausted; requested distribution impossible")
    report = {"schema": "original-task-generation-report.v1", "spec": spec.to_dict(),
        "generator_code_sha256": generator_code_sha256(), "validator_code_sha256": validator_code_sha256(),
        "validator_id": "independent-parser-fractions-reversal-floyd-warshall.v1",
        "family_counts": {k: dict(Counter(r["family"] for r in v)) for k, v in splits.items()},
        "rejected": dict(rejected), "split_policy": "sha256-canonical-task-mod10-train0to7-dev8-eval9.v1",
        "external_examples_used": False, "external_certification": False}
    return splits, report


def prepare_original_source(artifacts, evaluation_store, scanner, spec, *, experiment_id,
                            processing_profile_sha256):
    splits, report = generate_original_splits(spec)
    refs = {k: put_record(artifacts, {"schema": "original-task-examples.v1", "split": k, "examples": v})
            for k, v in splits.items() if k != "train"}
    protocol_ref = evaluation_store.freeze(experiment_id=experiment_id,
        evaluation_ref=refs["evaluation"], development_ref=refs["development"],
        method="precommitted exact-task correctness and JSON validity; deterministic greedy generation",
        limitations=["Internal prospective generated-task experiment, not independent certification.",
                     "Small correlated task families; no general language capability claim."])
    raw = b"".join(canonical_bytes(row) + b"\n" for row in splits["train"])
    source_ref = artifacts.put_bytes(raw, media_type="application/jsonl").artifact_ref
    files = {"original-tasks.jsonl": source_ref}
    generator_ref = put_record(artifacts, {"schema": "original-generator-provenance.v1", **report,
        "schema": "original-generator-provenance.v1", "code_sha256": generator_code_sha256(),
        "source_file_hashes": files, "development_ref": refs["development"], "evaluation_ref": refs["evaluation"]})
    rights_ref = put_record(artifacts, {"schema": "data-rights-evidence.v1", "license": "cc0-1.0",
        "training_permitted": True, "source_file_hashes": files, "generator_ref": generator_ref,
        "basis": "Original authored task grammar and computed mathematical/symbolic facts; no harvested examples."})
    candidate = DatasetCandidate("zz-original-tasks-" + source_ref[-20:], "original_task_registry",
        "hephaestus/original-verifiable-tasks", revision=source_ref, license="cc0-1.0",
        task_types=list(spec.families), domains=["math", "structured_output", "symbolic"], languages=["en"],
        estimated_rows=len(splits["train"]), estimated_bytes=len(raw),
        format_profile={"record_format": "jsonl"}, provenance={"generator_ref": generator_ref, "content_hash": source_ref},
        metadata={"acquisition_files": list(files), "synthetic": True}, evidence_refs=[generator_ref])
    scan = scanner.scan(files, protocol_ref=protocol_ref, processing_profile_sha256=processing_profile_sha256)
    review = InternalDatasetReview(candidate_sha256(candidate), files, processing_profile_sha256,
        scanner.tokenizer_sha256, protocol_ref, spec.families[0], rights_ref, put_record(artifacts, scan), generator_ref)
    return candidate, put_record(artifacts, review.to_dict()), protocol_ref, generator_ref


@dataclass(frozen=True)
class OriginalTaskRegistryProvider:
    candidates: tuple
    provider_id: str = "original_task_registry"
    def search(self, request):
        return tuple(c for c in self.candidates if set(c.task_types) & set(request.capability_targets))


def materialize_authorized_tokens(admission, candidate, review_ref, **scope):
    authorization = admission.authorize(candidate, review_ref, **scope)
    review = read_verified_json(admission.artifacts, review_ref)
    scanner = admission.scanner
    rows, _ = scanner.load_rows(review["source_file_hashes"])
    records = []
    for index, row in enumerate(rows):
        validate_original_example(row)
        prefix, _ = scanner.token_ids(row["prompt"] + "\nAnswer: ")
        target, unknown = scanner.token_ids(row["target"])
        if not prefix or not target or unknown:
            raise ValueError("invalid original-task tokenizer boundaries")
        records.append({"record_kind": FORMAT, "source_index": index,
            "sample_sha256": "sha256:" + sha256(canonical_bytes(row)).hexdigest(),
            "input_ids": prefix + target, "labels": [-100] * len(prefix) + target,
            "prefix_tokens": len(prefix)})
    raw = b"".join(canonical_bytes(row) + b"\n" for row in records)
    token_ref = admission.artifacts.put_bytes(raw, media_type="application/jsonl").artifact_ref
    return put_record(admission.artifacts, {"schema": FORMAT, "authorization_ref": authorization,
        "review_ref": review_ref, "source_file_hashes": review["source_file_hashes"],
        "tokenizer_sha256": scanner.tokenizer_sha256, "tokens_ref": token_ref,
        "rows": len(records), "tokens": sum(len(r["input_ids"]) for r in records),
        "maximum_width": max(len(r["input_ids"]) for r in records),
        "prefix_mask": -100, "eos_insertion": False, "truncation": False})


def load_authorized_tokens(admission, candidate, review_ref, materialized_ref, **scope):
    """Reproduce exact source/token/mask bytes at worker start and resume."""
    expected = materialize_authorized_tokens(admission, candidate, review_ref, **scope)
    if expected != materialized_ref:
        raise PermissionError("original-task materialization, authorization or tokenizer changed")
    record = read_verified_json(admission.artifacts, expected)
    with admission.artifacts.open(record["tokens_ref"]) as stream:
        return [json.loads(line) for line in stream.read().decode().splitlines()]
