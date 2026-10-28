"""Tier A machine authorization under explicit policy; no independent signature is minted.

Tier B never inherits this authority. Tier C continues through the unchanged
IndependentDatasetReviewPolicy and independent custody/signature verifiers.
"""
from dataclasses import dataclass
from hephaestus.schemas.internal_research import InternalResearchPolicy, InternalDatasetReview
from hephaestus.policy.dataset_review import candidate_sha256
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def put_record(artifacts, payload):
    return artifacts.put_bytes(canonical_bytes(payload), media_type="application/json").artifact_ref


@dataclass(frozen=True)
class InternalDatasetAdmissionPolicy:
    artifacts: object
    evaluation_store: object
    policy_ref: str
    trusted_policy_refs: frozenset[str]
    scanner: object = None

    def policy(self):
        if self.policy_ref not in self.trusted_policy_refs:
            raise PermissionError("internal policy is not preauthorized by composition owner")
        policy = InternalResearchPolicy(**read_verified_json(self.artifacts, self.policy_ref))
        policy.validate()
        self.evaluation_store.require_frozen(policy.experiment_id, policy.protocol_ref)
        return policy

    def require_clearance(self, candidate, review_ref, *, processing_profile_sha256,
                          tokenizer_sha256, experiment_id, workflow_id, run_id):
        policy = self.policy()
        if (experiment_id, workflow_id, run_id, processing_profile_sha256, tokenizer_sha256) != (
                policy.experiment_id, policy.workflow_id, policy.run_id,
                policy.processing_profile_sha256, policy.tokenizer_sha256):
            raise PermissionError("internal authorization experiment, processing or tokenizer scope mismatch")
        review = InternalDatasetReview(**read_verified_json(self.artifacts, review_ref))
        if (review.schema != "internal-dataset-review.v1" or
                review.candidate_sha256 != candidate_sha256(candidate) or
                review.processing_profile_sha256 != policy.processing_profile_sha256 or
                review.tokenizer_sha256 != policy.tokenizer_sha256 or
                review.protocol_ref != policy.protocol_ref or
                review.capability not in policy.allowed_capabilities or
                candidate.license not in policy.allowed_licenses):
            raise PermissionError("Tier B: source or internal review scope/rights unresolved")
        files = candidate.metadata.get("acquisition_files")
        if not files or set(files) != set(review.source_file_hashes) or len(files) != len(set(files)):
            raise ValueError("exact reviewed source file scope required")
        rights = read_verified_json(self.artifacts, review.rights_ref)
        if (rights.get("schema") != "data-rights-evidence.v1" or
                rights.get("license") != candidate.license or
                rights.get("source_file_hashes") != review.source_file_hashes or
                rights.get("training_permitted") is not True):
            raise PermissionError("data-specific rights evidence required; badges are insufficient")
        if review.generator_ref:
            generator = read_verified_json(self.artifacts, review.generator_ref)
            if (generator.get("schema") != "original-generator-provenance.v1" or
                    generator.get("code_sha256") not in policy.allowed_generator_code_sha256 or
                    generator.get("source_file_hashes") != review.source_file_hashes or
                    generator.get("external_examples_used") is not False or
                    rights.get("generator_ref") != review.generator_ref):
                raise PermissionError("original generator provenance not authorized")
            # A caller cannot paste an allowed code digest onto arbitrary data.
            # Reproduce original generation from committed version/spec/seed.
            from hephaestus.data.generated_factory import generate_original_splits, generator_code_sha256, validator_code_sha256
            from hephaestus.schemas.generated_data import GeneratedDataSpec
            import hashlib
            if (generator["code_sha256"] != generator_code_sha256() or
                    generator.get("validator_code_sha256") != validator_code_sha256()):
                raise PermissionError("original generator/validator code changed")
            reproduced, report = generate_original_splits(GeneratedDataSpec(**generator["spec"]))
            raw = b"".join(canonical_bytes(row) + b"\n" for row in reproduced["train"])
            if set(review.source_file_hashes.values()) != {"sha256:" + hashlib.sha256(raw).hexdigest()}:
                raise PermissionError("source is not the committed original generator output")
            protocol = self.evaluation_store.require_frozen(policy.experiment_id, policy.protocol_ref)
            for split, field in (("development", "development_ref"), ("evaluation", "evaluation_ref")):
                original = read_verified_json(self.artifacts, protocol[field])
                if original.get("examples") != reproduced[split] or generator.get(field) != protocol[field]:
                    raise PermissionError("original generated prospective split identity mismatch")
        elif review.rights_ref not in policy.trusted_rights_refs:
            raise PermissionError("original source rights evidence has not been accepted")
        scan = read_verified_json(self.artifacts, review.scan_ref)
        expected = {"schema": "internal-source-scan.v1", "coverage": "full_corpus",
            "source_file_hashes": review.source_file_hashes, "protocol_ref": policy.protocol_ref,
            "tokenizer_sha256": policy.tokenizer_sha256,
            "processing_profile_sha256": policy.processing_profile_sha256}
        if any(scan.get(k) != v for k, v in expected.items()):
            raise PermissionError("full-source scan is not bound to exact internal scope")
        for k in ("pii_flags", "malformed", "duplicates", "exact_overlap", "near_overlap", "semantic_overlap"):
            if type(scan.get(k)) is not int or scan[k] != 0:
                raise PermissionError("Tier B: source screening found unresolved " + k)
        for k, limit in (("rows", policy.maximum_rows), ("bytes", policy.maximum_bytes)):
            if type(scan.get(k)) is not int or not 0 < scan[k] <= limit:
                raise PermissionError("internal dataset resource budget exceeded")
        total, unknown = scan.get("total_tokens"), scan.get("unknown_tokens")
        if (type(total) is not int or type(unknown) is not int or total <= 0 or
                not 0 <= unknown <= total or unknown * 100 > total):
            raise PermissionError("actual tokenizer measurement failed")
        if (not scan.get("semantic_method") or not scan.get("limitations") or
                scan.get("external_certification") is not False):
            raise PermissionError("screening method and internal limitations required")
        # Hash and parse the actual full source now; review claims cannot substitute for bytes.
        import hashlib
        for ref in review.source_file_hashes.values():
            with self.artifacts.open(ref) as stream:
                raw = stream.read(policy.maximum_bytes + 1)
            if len(raw) > policy.maximum_bytes or "sha256:" + hashlib.sha256(raw).hexdigest() != ref:
                raise ValueError("source bytes differ from full-corpus review")
        if self.scanner is None:
            raise PermissionError("source scanner required; stored pass flags are not evidence")
        measured = self.scanner.scan(review.source_file_hashes, protocol_ref=policy.protocol_ref,
                                    processing_profile_sha256=policy.processing_profile_sha256)
        if measured != scan:
            raise PermissionError("source scan differs from independently recomputed measurements")
        return review

    def authorize(self, candidate, review_ref, **scope):
        review = self.require_clearance(candidate, review_ref, **scope)
        policy = self.policy()
        return put_record(self.artifacts, {"schema": "internal-dataset-authorization.v1",
            "authority": "preauthorized_internal_policy", "policy_ref": self.policy_ref,
            **scope, "candidate_sha256": candidate_sha256(candidate), "review_ref": review_ref,
            "source_file_hashes": review.source_file_hashes, "tier": "A",
            "limits": {k: getattr(policy, k) for k in ("maximum_bytes", "maximum_rows",
                "maximum_training_steps", "maximum_cpu_seconds")},
            "external_certification": False, "promotion_authorized": False})
