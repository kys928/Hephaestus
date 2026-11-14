# Bounded autonomous scientific investigation and self-repair (implementation v1)

## Precise target

The system can now **receive one supported natural-language research objective**,
retrieve its own independently reviewed source references, authenticate original
source bytes, conduct training-free evaluation, and record the raw frozen-role
inference receipts. For a separately verified code regression in an allowlisted
Diagnostician module, it can request a bounded code patch, reproduce red-to-green
in an external offline sandbox and publish a **draft** review PR.

This is an **opt-in capability**, not a claim that the entire original 45-stage
training/autonomy controller is production-ready. No code automatically merges,
no running process reloads its patched source, and no paid model experiment is
implicitly authorized.

## How the objectives enter Hephaestus

`research.durable_jobs.submit_research_objective` takes a **natural-language
objective**, immutable run/lineage identity, and the existing `JobQueue`. The
request is stored by SHA-256 and submitted under a stable idempotency key.

`ResearchObjectiveJobHandler` is a callable for the existing
`jobs.DurableWorker`; it reopens the original immutable request, checks that
the kind and authority flags are unchanged, and chooses from a finite set of
registered capabilities:

- `diagnosis_evaluation_only`: the original-frozen Diagnostician evidence
  pilot, with the exact original source identities and an injected real frozen
  role observer, no training stage.
- `diagnosis_code_repair`: one reviewed Diagnostician coverage-probe module,
  one unchanged regression suite, sandboxed model-generated replacement, draft
  GitHub review PR.

Unknown/free-form goals fail closed. Natural-language matching is a **bounded
capability router**, not general research intelligence or a new unrestricted
agent. New research classes require a separately reviewed domain contract.

The default infrastructure `worker --handler` requires a deployment-provided
callable. The production composition of Docker, coder model, GitHub PR token,
and GPU-qualified frozen-role generator is **not automatically enabled** by
installing this package. An operator must still deploy the appropriate
trusted adapter and ensure environment/source admission.

## Evidence acquisition

`SourceCatalog` admits a reviewed JSON index only if its raw bytes equal an
**independently pinned expected SHA-256**, supplied by the trusted deployment,
not the model or catalog itself. For the first catalog:

- `configs/research/diagnosis_historical_source_catalog_v1.json`
- independent CI-observed catalog digest:
  `sha256:657dc419dfb67f82b882d33f8ac407f376250cdb1a3a10cb0fe64c398b3d7762`
- original raw S3 SHA:
  `sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
- controlled successor raw S3 SHA:
  `sha256:bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`

The reader calls **GET on exact allowlisted RunPod S3 keys only**; it does not
enumerate storage, execute remote code or write back. Source bytes are size,
content-hash and JSONL-count checked before any source-level measurement.
The v2 coverage probe separately reports embedded marker occurrences,
nonempty complete wrappers, parseable structured targets and the frozen
task-coverage gap. Original S3 data is not committed to Git.

The first adapter searches one controlled, pre-reviewed two-source family.
That is intentionally **not yet general web research or arbitrary S3
discovery**. Unreviewed source keys cannot become scientific ground truth
merely because a language model found them.

## No-training experimental execution

`EvaluationOnlyResearchJob` calls a seeded `FrozenRoleObserver` for each
admitted source state, using actual source-observation artifact references.
`CertifiedDiagnosisObserver` composes the already existing frozen
`CertifiedRoleReasoner`, `RoleTask`, `EvidenceRegistry` and original
inference receipt custody. A caller that supplies no real observer blocks;
it never falls back to a fake model.

Reports preserve decision distributions, exact case source digests, seed
identities, original model receipt references and source-measurement differences.
They explicitly set `training_performed=false`,
`independent_scientific_certification=false` and do not treat a change in
instruction wrappers as an original Granite failure.

The standard eight-phase/45-step **training** spine is unchanged. This is a
separate queued *evaluation-only* research job using the same durable queue
and artifact custody, not a fictional completed training campaign or a
replacement campaign scheduler. End-to-end genuine frozen-role GPU inference
requires fresh scoped compute approval, model availability and a deployment
composition; that experiment has **not** been executed.

## Bounded self-repair

The only automatic code-repair surface in this implementation is:

- source: `src/hephaestus/diagnosis/probes.py` (the vetted registry target)
- frozen unchanged regression suite:
  `tests/test_targeted_post_failure_probes.py`
- no arbitrary shell, new tests, secret files, policy files, control spine,
  checkpoint loader, source or model weights, frozen evaluation pack or
  deployed activation.

`AutonomousDiagnosisRepairEpisode` obtains a clean exact Git commit; then
tests a *copied* baseline using `OfflineDockerPytestRunner` before asking a
separate local coder for a JSON source candidate. The coder is text-only
(`LoopbackOllamaPatchModel` supports the fixed loopback Ollama endpoint).
A bounded `PatchCandidate` must match the expected source hash and file
path, compile as Python, and stay within a 100-line change ceiling.

Both baseline and candidate run in a prebuilt digest-pinned Docker image:
read-only source mount, network disabled, all capabilities dropped, no new
privileges, unprivileged UID, resource limits and no host secrets. Only a
**real baseline failure followed by a candidate pass** is reviewable.
The initial acceptance demonstrated red-to-green on an intentionally failing
fixture; this is engineering evidence, not production self-repair success.

`GitHubReviewBranchPublisher` can create a new diagnosis repair branch and
**draft PR**, with exact remote main commit guard and source blob recheck.
Its method set contains no GitHub merge request or direct main write. The
publisher requires a separately provisioned scoped GitHub token; do not give
the model that token. A later independent human review is still required
before merging executable changes.

## Supported automatic continuation vs unresolved work

Implemented: durable job intake, one-shot evidence-only study, source
authentication, scoped red-to-green regression checking, PR proposal, safe
no-regression stop, duplicate objective suppression and explicit failure
of unknown or unconfigured capabilities.

**Still missing before claiming full autonomous self-repair**:

1. **General objective planning:** only two bounded Diagnostician capabilities
   are currently routable from natural language, not unrestricted research.
2. **General evidence discovery:** the first immutable reviewed S3 catalog is
   a single matched source pair; no universal source-search/index builder,
   automatic external adjudication or protected holdout oracle exists.
3. **Live model/compute deployment:** neither the genuine frozen-role GPU
   pilot nor the separate local coder/PR publication has been operated here
   with real weights/credentials. CPU mocks and real Docker acceptance
   cannot prove the model will identify a repair.
4. **General next-experiment transitions:** the original
   `StagedActionIntentService` still blocks default
   branch/restart/rollback/stage/policy next-run handoffs. This first
   code-repair route creates a **review PR**, not a new training lineage.
5. **Promotion and rollout:** no autonomous merge, restarts, rollback of
   active source, independent scientific certification or new GPU spend.
   These require separate governance and bounded deployment validation.

No model claim can override hashes, identities, approvals, trusted source
review or independent evaluation. The precise acceptance result is a
**bounded self-repair mechanism**, not unrestricted autonomous scientific
software engineering.
