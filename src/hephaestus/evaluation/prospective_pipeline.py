"""One-use custodian-side CPU generation, raw custody, analysis and cold readback.

Trusted public keys must be configured by a genuinely separate authority. This
module verifies signatures; it never creates a production key or certificate.
The training controller should not run this service with access to the holdout.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
from importlib.metadata import version

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.schemas.prospective_custody import CustodyGrant, DevelopmentAttempt, CustodySession
from hephaestus.state.prospective_session_store import ProspectiveSessionStore, STUDY_SESSIONS
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.training.cpu_chat_json_recipe import chat_prompt
from .exact_json import SCORER_VERSION, score_prediction, strict_json_object
from .paired_holdout import paired_binary_statistics
from .prospective_design import holdout_commitment, validate_holdout_distribution

ORIGINAL_WEIGHTS = "sha256:5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
ORIGINAL_TOKENIZER = "sha256:9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"


def digest(value):
    return "sha256:" + sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def generation_profile():
    from . import exact_json, paired_holdout, prospective_design
    from hephaestus.training import cpu_chat_json_recipe
    return {"schema": "prospective-cpu-execution-profile.v1",
        "frameworks": {name: version(name) for name in ("torch", "transformers", "peft", "tokenizers")},
        "code": {name: file_digest(path) for name, path in {
            "pipeline": __file__, "scorer": exact_json.__file__, "statistics": paired_holdout.__file__,
            "design": prospective_design.__file__, "chat_recipe": cpu_chat_json_recipe.__file__}.items()}}


def select_development_seed(attempts, design):
    for row in attempts:
        DevelopmentAttempt.from_dict(row)
    if ({r["seed"] for r in attempts} != set(design.training_seeds)
            or len(attempts) != len(design.training_seeds)):
        raise ValueError("all predeclared seed attempts, including failures, are required")
    completed = [r for r in attempts if r.get("status") == "completed"]
    if any(r.get("status") not in {"completed", "failed"} for r in attempts) or not completed:
        raise ValueError("no complete development selection or unresolved attempt")
    for row in completed:
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("seed", "development_correct", "development_valid_json")):
            raise ValueError("development selection requires exact nonnegative counts")
    return max(completed, key=lambda r: (r["development_correct"], r["development_valid_json"], -r["seed"]))["seed"]


def verify_custodian_grant(envelope, *, trusted_public_keys, expected, verification_time=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        payload = envelope["payload"]
        CustodyGrant.from_dict(payload)
        if (set(envelope) != {"payload", "signature_hex"}
                or set(payload) != set(expected) | {"authority_id", "issued_at", "expires_at"}
                or any(payload.get(k) != v for k, v in expected.items())
                or type(payload.get("allowed_sessions")) is not int or payload["allowed_sessions"] != 1):
            raise ValueError("grant scope differs")
        now = verification_time or datetime.now(timezone.utc)
        issued = datetime.fromisoformat(payload["issued_at"])
        expires = datetime.fromisoformat(payload["expires_at"])
        if issued.utcoffset() is None or expires.utcoffset() is None or not issued <= now < expires:
            raise ValueError("custody grant expired, future or timezone-naive")
        public = bytes.fromhex(trusted_public_keys[payload["authority_id"]])
        Ed25519PublicKey.from_public_bytes(public).verify(
            bytes.fromhex(envelope["signature_hex"]), canonical_bytes(payload))
    except Exception as exc:
        raise PermissionError("custodian grant is not trusted, current and exactly scoped") from exc
    return digest(envelope)


def freeze_development_selection(artifacts, design, rows, bindings, *, failed_attempts=()):
    """Compute selection from actual development generations, before final seal."""
    if design.purpose == "confirmatory" and digest({"schema": "original-task-examples.v1",
            "split": "development", "examples": rows}) != "sha256:4c9d54866719545f9a54466cd6da7b9aed4b79f935160d61e42e0639675c3f1f":
        raise PermissionError("original development bytes have not been recovered")
    attempts, outputs = list(failed_attempts), {}
    for binding in bindings:
        if binding.seed is None:
            continue
        output = generate_cpu(binding, rows, design)
        outputs[binding.label] = output
        scored = [score_prediction(s["answer_text"], r["target"])
                  for r, s in zip(rows, output["samples"], strict=True)]
        attempts.append({"seed": binding.seed, "status": "completed",
            "development_correct": sum(s["exact_correct"] for s in scored),
            "development_valid_json": sum(s["valid_json"] for s in scored)})
    record = {"schema": "prospective-development-selection.v1", "rows": rows,
              "outputs": outputs, "attempts": attempts}
    ref = artifacts.put_bytes(canonical_bytes(record)).artifact_ref
    return {"attempts": attempts, "selected_seed": select_development_seed(attempts, design),
            "development_evidence_ref": ref}


def verify_development_selection(artifacts, selection, models, exposure_ids, design):
    evidence = read_verified_json(artifacts, selection["development_evidence_ref"])
    if evidence["attempts"] != selection["attempts"]:
        raise PermissionError("development attempt ledger differs from immutable evidence")
    rows = evidence["rows"]
    holdout_commitment(rows)
    if (not rows or len({r["sample_id"] for r in rows}) != len(rows)
            or any(r["sample_id"] not in exposure_ids for r in rows)):
        raise PermissionError("development examples must be unique and in the exposure ledger")
    if design.purpose == "confirmatory" and digest({"schema": "original-task-examples.v1",
            "split": "development", "examples": rows}) != "sha256:4c9d54866719545f9a54466cd6da7b9aed4b79f935160d61e42e0639675c3f1f":
        raise PermissionError("development selection used a different source distribution")
    for attempt in selection["attempts"]:
        if attempt["status"] != "completed":
            continue
        label = f"seed-{attempt['seed']}"
        output = evidence["outputs"][label]
        expected_identity = models[label].identity() if isinstance(models[label], ProspectiveModelBinding) else models[label]
        if output["model_identity"] != expected_identity:
            raise PermissionError("development-selected checkpoint was replaced")
        if [r["sample_id"] for r in rows] != [s["sample_id"] for s in output["samples"]]:
            raise PermissionError("development output/example pairing differs")
        scored = [score_prediction(s["answer_text"], r["target"])
                  for r, s in zip(rows, output["samples"], strict=True)]
        if (attempt["development_correct"] != sum(s["exact_correct"] for s in scored)
                or attempt["development_valid_json"] != sum(s["valid_json"] for s in scored)):
            raise PermissionError("development scores do not match original outputs")


def verify_model_binding(binding: ProspectiveModelBinding, design: ProspectiveDesign):
    for path, expected in ((binding.model_path, binding.model_identity),
                           (binding.tokenizer_path, binding.tokenizer_identity)):
        if directory_content_identity(path) != expected:
            raise PermissionError("sealed model/tokenizer identity changed")
    if binding.adapter_path is not None:
        if directory_content_identity(binding.adapter_path) != binding.adapter_identity:
            raise PermissionError("sealed adapter identity changed")
    if binding.seed is not None:
        if (binding.seed not in design.training_seeds or binding.label != f"seed-{binding.seed}"
                or not binding.job_path or file_digest(binding.job_path) != binding.job_hash):
            raise PermissionError("candidate is missing its original seed/job identity")
        job = strict_json_object(Path(binding.job_path).read_text())
        if job is None or job["training_config"]["seed"] != binding.seed:
            raise PermissionError("candidate seed differs from actual prepared job")
        if (not binding.adapter_path or not binding.checkpoint_path
                or Path(binding.adapter_path).resolve() != (Path(binding.checkpoint_path) / "model").resolve()
                or job["compatibility"]["model_revision"] != binding.model_identity
                or job["compatibility"]["tokenizer_revision"] != binding.tokenizer_identity):
            raise PermissionError("candidate adapter/base/tokenizer differs from its checkpoint job")
        valid, manifest = validate_checkpoint_manifest(binding.checkpoint_path or "")
        if not valid or manifest != binding.checkpoint_manifest_hash:
            raise PermissionError("candidate checkpoint differs from retained training commitment")
        import torch
        state = torch.load(Path(binding.checkpoint_path) / "training_state.pt", map_location="cpu", weights_only=True)
        if (state["config_fingerprint"] != job["config_fingerprint"]
                or state["step"] < 1 or state["compatibility"] != job["compatibility"]):
            raise PermissionError("candidate optimization state is incomplete or unrelated")
        if design.purpose == "confirmatory":
            cfg = job["training_config"]
            recipe = cfg.get("peft_config", {})
            if (state["step"] != 96 or state["examples_processed"] != 96
                    or cfg["learning_rate"] != .0005 or recipe.get("rank") != 8
                    or recipe.get("alpha") != 16 or set(recipe.get("target_modules", [])) != {"q_proj", "v_proj"}):
                raise PermissionError("candidate differs from the predeclared 96-update LoRA recipe")
            from transformers import AutoTokenizer
            from hephaestus.data.generated_factory import generate_original_splits, generator_code_sha256
            from hephaestus.schemas.generated_data import GeneratedDataSpec
            from hephaestus.schemas.prompt_target_supervision import PromptTargetSupervision
            from hephaestus.data.chunking import supervision_segments
            if generator_code_sha256() != "sha256:1923382f27a35c3826a9d81623200533b22b8f1ea35142d5f6c1bcaae0cad101":
                raise PermissionError("original training source generator has drifted")
            original = generate_original_splits(GeneratedDataSpec(20261028))[0]["train"]
            if "sha256:" + sha256(b"".join(canonical_bytes(r)+b"\n" for r in original)).hexdigest() != (
                    "sha256:602d8fb5a59d2db02cd6a9239a98096ef55c0d906ed7307c1e15f275ec2fb68d"):
                raise PermissionError("original training source reconstruction differs")
            by_family = {f: [r for r in original if r["family"] == f] for f in design.families}
            expected = [by_family[f][i] for i in range(24) for f in design.families]
            data_path = job["compatibility"]["processed_dataset_ref"]
            if file_digest(data_path) != job["compatibility"]["processed_dataset_hash"]:
                raise PermissionError("training input bytes differ from original prepared job")
            processed = [strict_json_object(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
            if len(processed) != 96:
                raise PermissionError("training data must be exactly one complete original epoch")
            tokenizer = AutoTokenizer.from_pretrained(binding.tokenizer_path, local_files_only=True, trust_remote_code=False)
            for row, record in zip(expected, processed, strict=True):
                supervision = PromptTargetSupervision.from_dict(record["supervision"])
                prefix, target = supervision_segments(supervision)
                if (prefix != chat_prompt(tokenizer, row["prompt"])
                        or target != row["target"] + tokenizer.eos_token or record["text"] != prefix+target):
                    raise PermissionError("training chat/target/source order differs from predeclared original recipe")
    elif binding.label != "baseline" or binding.adapter_path is not None:
        raise PermissionError("baseline binding must be the unadapted original model")
    if design.purpose == "confirmatory":
        if (file_digest(Path(binding.model_path) / "model.safetensors") != ORIGINAL_WEIGHTS
                or file_digest(Path(binding.tokenizer_path) / "tokenizer.json") != ORIGINAL_TOKENIZER):
            raise PermissionError("confirmatory original model/tokenizer bytes are not recovered")


def generate_cpu(binding, rows, design):
    """Only this concrete loader produces real-generation receipts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    verify_model_binding(binding, design)
    tokenizer = AutoTokenizer.from_pretrained(binding.tokenizer_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(binding.model_path, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.float32).to("cpu")
    if binding.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, binding.adapter_path, local_files_only=True, is_trainable=False)
    model.eval()
    eos = tokenizer.eos_token_id
    if type(eos) is not int or not tokenizer.chat_template:
        raise ValueError("pinned EOS and chat template required")
    outputs = []
    for row in rows:
        prefix = chat_prompt(tokenizer, row["prompt"])
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if len(ids) + design.max_new_tokens > int(model.config.max_position_embeddings):
            raise ValueError("frozen generation prompt cannot be truncated")
        tensor = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            result = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                do_sample=False, num_beams=1, max_new_tokens=design.max_new_tokens,
                min_new_tokens=0, min_length=0, repetition_penalty=1.0,
                eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                forced_eos_token_id=None, forced_bos_token_id=None)
        new = result[0, len(ids):].tolist()
        terminated = bool(new and new[-1] == eos)
        body = new[:-1] if terminated else new
        outputs.append({"sample_id": row["sample_id"], "prompt_tokens": ids,
            "generated_tokens": new, "eos_token_id": eos, "eos": terminated,
            "truncated": len(new) >= design.max_new_tokens and not terminated,
            "raw_decode": tokenizer.decode(new, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "answer_text": tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "prompt_sha256": "sha256:" + sha256(prefix.encode()).hexdigest()})
    verify_model_binding(binding, design)
    return {"schema": "prospective-real-cpu-generation.v1", "model_identity": binding.identity(),
        "execution_profile": generation_profile(),
        "chat_template_sha256": "sha256:" + sha256(tokenizer.chat_template.encode()).hexdigest(),
        "scorer_version": SCORER_VERSION, "samples": outputs, "execution_device": "cpu"}


def analyse_pairs(rows, baseline, candidate, design, *, alpha=.05):
    if (baseline["chat_template_sha256"] != candidate["chat_template_sha256"]
            or baseline["scorer_version"] != SCORER_VERSION or candidate["scorer_version"] != SCORER_VERSION):
        raise ValueError("generation template/scorer parity failed")
    before, after, families, fmt = [], [], {}, [0, 0]
    eos_failures, truncations = 0, 0
    if len(baseline["samples"]) != len(rows) or len(candidate["samples"]) != len(rows):
        raise ValueError("incomplete paired generation")
    for row, b, c in zip(rows, baseline["samples"], candidate["samples"], strict=True):
        if row["sample_id"] != b["sample_id"] or row["sample_id"] != c["sample_id"] or b["prompt_tokens"] != c["prompt_tokens"]:
            raise ValueError("generation sample/input parity failed")
        for receipt in (b, c):
            tokens = receipt["generated_tokens"]
            if (not isinstance(tokens, list) or not 1 <= len(tokens) <= design.max_new_tokens
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or type(receipt["eos_token_id"]) is not int
                    or type(receipt["eos"]) is not bool or type(receipt["truncated"]) is not bool):
                raise ValueError("invalid actual generation token/termination receipt")
            observed_eos = bool(tokens and tokens[-1] == receipt["eos_token_id"])
            if (receipt["eos"] != observed_eos or receipt["truncated"] != (
                    len(tokens) >= design.max_new_tokens and not observed_eos)):
                raise ValueError("EOS/truncation flags differ from actual token receipt")
        bs = score_prediction(b["answer_text"], row["target"])
        cs = score_prediction(c["answer_text"], row["target"])
        before.append(bs["exact_correct"]); after.append(cs["exact_correct"])
        fmt[0] += bs["valid_json"]; fmt[1] += cs["valid_json"]
        families.setdefault(row["family"], []).append(int(cs["exact_correct"])-int(bs["exact_correct"]))
        eos_failures += not c["eos"]; truncations += c["truncated"]
    effect = paired_binary_statistics(before, after, bootstrap_seed=design.bootstrap_seed,
                                      bootstrap_draws=design.bootstrap_draws)
    rng = random.Random(design.bootstrap_seed)
    estimates = sorted(sum(sum(rng.choice(values) for _ in values) for values in families.values())/len(rows)
                       for _ in range(design.bootstrap_draws))
    lower = estimates[int(alpha/2 * design.bootstrap_draws)]
    upper = estimates[min(design.bootstrap_draws-1, int((1-alpha/2)*design.bootstrap_draws))]
    effect["stratified_paired_interval"] = [lower, upper]
    effect["interval_confidence"] = 1-alpha
    gates = {"gain_at_least_5pp": effect["paired_absolute_gain"] >= design.minimum_gain,
        "exact_mcnemar": effect["exact_mcnemar_two_sided_p"] < alpha,
        "stratified_lower_positive": lower > 0, "nonregressing_format": fmt[1] >= fmt[0],
        "all_families_nonnegative": all(sum(v) >= 0 for v in families.values()),
        "losses_at_most_one_percent": effect["losses"] <= len(rows)//100,
        "all_candidate_eos": eos_failures == 0, "zero_candidate_truncation": truncations == 0}
    return {"paired_exact": effect, "gates": gates, "all_gates_pass": all(gates.values()),
        "family_gains": {k: sum(v)/len(v) for k, v in families.items()},
        "eos_failures": eos_failures, "truncations": truncations}


class ProspectiveCustodyRunner:
    def __init__(self, repository: StateRepository, artifacts: ArtifactStore, trusted_public_keys: dict):
        self.repository, self.artifacts, self.trusted_public_keys = repository, artifacts, trusted_public_keys
        self.sessions = ProspectiveSessionStore(repository)

    def run(self, *, study_id, design, rows, exposure_sample_ids, bindings, selection, grant):
        design.commitment()
        models = {b.label: b for b in bindings}
        if len(models) != len(bindings) or "baseline" not in models:
            raise ValueError("unique baseline and candidate model bindings required")
        selected = select_development_seed(selection["attempts"], design)
        if selection.get("selected_seed") != selected or f"seed-{selected}" not in models:
            raise ValueError("candidate was not selected solely by the precommitted development rule")
        expected_labels = {"baseline"} | {f"seed-{r['seed']}" for r in selection["attempts"] if r["status"] == "completed"}
        if set(models) != expected_labels:
            raise ValueError("all completed predeclared seeds must be evaluated in the same session")
        if any(b.model_identity != models["baseline"].model_identity
               or b.tokenizer_identity != models["baseline"].tokenizer_identity for b in bindings):
            raise PermissionError("baseline/candidate base model and tokenizer must match exactly")
        verify_development_selection(self.artifacts, selection, models, exposure_sample_ids, design)
        if (Counter(r["family"] for r in rows) != Counter({f: design.cases_per_family for f in design.families})
                or not exposure_sample_ids or any(r["sample_id"] in exposure_sample_ids for r in rows)):
            raise ValueError("holdout balance/exposure contract failed")
        rows = sorted(rows, key=lambda r: r["sample_id"])
        validate_holdout_distribution(rows, design)
        submission = {"study_id": study_id, "design_commitment": design.commitment(),
            "execution_profile": generation_profile(),
            "models": {k: b.identity() for k, b in models.items()}, "selection": selection,
            "holdout_commitment": holdout_commitment(rows),
            "exposure_commitment": digest(sorted(exposure_sample_ids)), "scorer_version": SCORER_VERSION}
        expected = {"study_id": study_id, "purpose": design.purpose, "allowed_sessions": 1,
                    "submission_commitment": digest(submission)}
        with execution_owner(self.repository, "prospective-study", study_id):
            prior = self.sessions.get(study_id)
            if prior is not None:
                if prior["submission_commitment"] != digest(submission):
                    raise PermissionError("study identity cannot be reused for changed models, selection or holdout")
                if prior["status"] == "completed":
                    if digest(grant) != prior["grant_ref"]:
                        raise PermissionError("completed session authority cannot be replaced")
                    verify_custodian_grant(grant, trusted_public_keys=self.trusted_public_keys, expected=expected,
                        verification_time=datetime.fromisoformat(prior["started_at"]))
                    return read_verified_json(self.artifacts, prior["bundle_ref"])
                raise PermissionError("holdout session already consumed; independent reconciliation required")
            grant_ref = verify_custodian_grant(grant, trusted_public_keys=self.trusted_public_keys, expected=expected)
            for binding in bindings:
                verify_model_binding(binding, design)
            started_at = datetime.now(timezone.utc).isoformat()
            verify_custodian_grant(grant, trusted_public_keys=self.trusted_public_keys, expected=expected,
                                   verification_time=datetime.fromisoformat(started_at))
            session = CustodySession(study_id, digest(submission),
                self.artifacts.put_bytes(canonical_bytes(submission)).artifact_ref,
                self.artifacts.put_bytes(canonical_bytes(grant), expected_hash=grant_ref).artifact_ref, started_at)
            self.sessions.consume(session)
            refs = {}
            try:
                for label, binding in models.items():
                    output = generate_cpu(binding, rows, design)
                    refs[label] = self.artifacts.put_bytes(canonical_bytes(output)).artifact_ref
                generated = {k: read_verified_json(self.artifacts, ref) for k, ref in refs.items()}
                primary = analyse_pairs(rows, generated["baseline"], generated[f"seed-{selected}"], design)
                replications = {str(seed): analyse_pairs(rows, generated["baseline"], generated[f"seed-{seed}"],
                    design, alpha=design.alpha/3) for seed in design.training_seeds if f"seed-{seed}" in generated}
                bundle = {"schema": "prospective-custody-bundle.v1", "purpose": design.purpose,
                    "submission": submission, "design": design.to_dict(), "grant": grant,
                    "session_started_at": started_at,
                    "holdout_ref": self.artifacts.put_bytes(canonical_bytes({"rows": rows})).artifact_ref,
                    "exposure_ref": self.artifacts.put_bytes(canonical_bytes({"sample_ids": sorted(exposure_sample_ids)})).artifact_ref,
                    "generation_refs": refs, "primary": primary, "replications": replications,
                    "selected_procedure_effect_supported": design.purpose == "confirmatory" and primary["all_gates_pass"],
                    "three_observed_seed_replications": design.purpose == "confirmatory" and len(replications) == 3
                        and all(r["all_gates_pass"] for r in replications.values()),
                    "external_certification": False, "model_promoted": False,
                    "limitations": ["Entrusted authority independence is an out-of-band requirement.",
                        "A signed custody grant is not a historical external scientific certificate.",
                        "Claims are conditional on four fixed task families/templates."]}
                bundle_ref = self.artifacts.put_bytes(canonical_bytes(bundle)).artifact_ref
                session.status, session.bundle_ref = "completed", bundle_ref
                self.sessions.finish(session)
                return read_verified_json(self.artifacts, bundle_ref)
            except BaseException as exc:
                # Exposure cannot be undone. Retain partial output references and
                # never allow a failed/in-flight session to tune/retry on holdout.
                prior = self.sessions.get(study_id)
                if prior is not None and prior["status"] == "consumed":
                    session.status, session.error_type = "failed", type(exc).__name__
                    session.partial_generation_refs = refs
                    self.sessions.finish(session)
                raise


def verify_cold_bundle(bundle, artifacts, *, expected_bundle_ref, trusted_public_keys, bindings=None, reexecute=False):
    if digest(bundle) != expected_bundle_ref or read_verified_json(artifacts, expected_bundle_ref) != bundle:
        raise ValueError("cold bundle differs from the separately retained publication commitment")
    design = ProspectiveDesign.from_dict(bundle["design"])
    submission = bundle["submission"]
    if (bundle["purpose"] != design.purpose or submission["design_commitment"] != design.commitment()
            or submission["scorer_version"] != SCORER_VERSION
            or bundle["external_certification"] is not False or bundle["model_promoted"] is not False):
        raise ValueError("cold design, scorer or authority claim differs from its scope")
    verify_custodian_grant(bundle["grant"], trusted_public_keys=trusted_public_keys,
        verification_time=datetime.fromisoformat(bundle["session_started_at"]), expected={
        "study_id": submission["study_id"], "purpose": design.purpose, "allowed_sessions": 1,
        "submission_commitment": digest(submission)})
    rows = read_verified_json(artifacts, bundle["holdout_ref"])["rows"]
    validate_holdout_distribution(rows, design)
    exposure = set(read_verified_json(artifacts, bundle["exposure_ref"])["sample_ids"])
    if (not exposure or digest(sorted(exposure)) != submission["exposure_commitment"]
            or any(r["sample_id"] in exposure for r in rows)
            or Counter(r["family"] for r in rows) != Counter({f: design.cases_per_family for f in design.families})):
        raise ValueError("cold balance or exposure ledger differs")
    if holdout_commitment(rows) != submission["holdout_commitment"]:
        raise ValueError("cold holdout commitment differs")
    generated = {k: read_verified_json(artifacts, ref) for k, ref in bundle["generation_refs"].items()}
    if (set(generated) != set(submission["models"])
            or select_development_seed(submission["selection"]["attempts"], design) != submission["selection"]["selected_seed"]):
        raise ValueError("cold model set or predeclared development selection differs")
    verify_development_selection(artifacts, submission["selection"], submission["models"], exposure, design)
    for label, output in generated.items():
        if (output["model_identity"] != submission["models"][label]
                or output["execution_profile"] != submission["execution_profile"]):
            raise ValueError("cold generation belongs to another model")
    if bindings is not None:
        from transformers import AutoTokenizer
        if generation_profile() != submission["execution_profile"]:
            raise ValueError("cold execution profile differs; replay with the pinned source/environment")
        if {b.label: b.identity() for b in bindings} != submission["models"]:
            raise ValueError("cold model bindings differ from sealed submission")
        for binding in bindings:
            verify_model_binding(binding, design)
            tokenizer = AutoTokenizer.from_pretrained(binding.tokenizer_path, local_files_only=True, trust_remote_code=False)
            if generated[binding.label]["chat_template_sha256"] != "sha256:" + sha256(tokenizer.chat_template.encode()).hexdigest():
                raise ValueError("cold chat template hash differs from pinned tokenizer")
            for row, sample in zip(rows, generated[binding.label]["samples"], strict=True):
                ids = sample["generated_tokens"]
                body = ids[:-1] if sample["eos"] else ids
                if (sample["eos_token_id"] != tokenizer.eos_token_id
                        or sample["prompt_tokens"] != tokenizer.encode(chat_prompt(tokenizer, row["prompt"]), add_special_tokens=False)
                        or sample["answer_text"] != tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        or sample["raw_decode"] != tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)):
                    raise ValueError("cold token decoding/prompt receipt differs from original text")
            if reexecute and generate_cpu(binding, rows, design) != generated[binding.label]:
                raise ValueError("cold actual model reexecution differs")
    selected = submission["selection"]["selected_seed"]
    primary = analyse_pairs(rows, generated["baseline"], generated[f"seed-{selected}"], design)
    replications = {str(seed): analyse_pairs(rows, generated["baseline"], generated[f"seed-{seed}"],
        design, alpha=design.alpha/3) for seed in design.training_seeds if f"seed-{seed}" in generated}
    if primary != bundle["primary"] or replications != bundle["replications"]:
        raise ValueError("cold raw-output reanalysis differs")
    if (bundle["selected_procedure_effect_supported"] != (design.purpose == "confirmatory" and primary["all_gates_pass"])
            or bundle["three_observed_seed_replications"] != (design.purpose == "confirmatory" and len(replications) == 3
                and all(r["all_gates_pass"] for r in replications.values()))):
        raise ValueError("cold scientific claim flags differ from actual gates")
    return {"verified": True, "purpose": design.purpose, "raw_outputs_reanalysed": True,
            "token_decoding_verified": bindings is not None, "model_reexecution_verified": bindings is not None and reexecute,
            "external_independent_certification": False, "model_promoted": False}
