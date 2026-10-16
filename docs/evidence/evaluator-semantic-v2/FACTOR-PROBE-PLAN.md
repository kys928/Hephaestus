# Evaluator D step-96 — factor-only inference probe protocol (not yet executed)

**Status:** CPU protocol validated, **paid inference not authorized**. This change does not contain or trigger a paid GPU launcher. It is an implementation and scientific preflight, not proof that the model can report the six factors.

## Scientific question

On the same evidence where D produced an incorrect final scientific decision, can the exact **saved D step-96 adapter** independently report the six underlying factors correctly?

Primary questions:
- Does it report `provenance_valid=True` when the frozen evidence says the artifact is valid, especially when the saved decision is incorrectly `inconclusive`?
- Can it distinguish `variance_stable=False` from invalid provenance, especially for `recheck_required` cases?
- Does it recognize `effect=neutral` within the frozen practical-equivalence threshold?
- Are reported-factor errors associated with Planner advisory presence?

## Protocol

The CPU-only [preflight workflow](https://github.com/kys928/Hephaestus/actions/runs/38960467381) validated **64 fixed prompts, eight per state** (30 relevant unit tests passed). Cases are *exactly* the previous D balanced development screen, with the original frozen diagnostic SHA and selection SHA. This enables paired comparison with **persisted independent D decisions** without paying for another 64 state-decision predictions.

The six-field factor-reporting prompt contains the verified evidence, verified machine state and advisory input, but **no scientific-state label, gold factor vector, target rationale or certification instruction**. JSON outputs must have exactly:
`gate_pass, complete, provenance_valid, effect, variance_stable, recheck_satisfied`.

The future report must include the **unchanged D step-96 adapter SHA**, case ID and SHA256 of the exact factor-only prompt. Scoring checks those identities, rejects missing and duplicate cases, strictly parses JSON without extraction from prose, and compares each factor with independently decoded frozen reference evidence. It also compares the saved D scientific decision against the oracle applied to the newly reported factors.

Outcome categories:
- **Factor reports correct, saved decision wrong:** observable factor/precedence mismatch across *different prompts*, a strong reason to test precedence directly.
- **Factor reports wrong, saved decision follows oracle(report):** observable factor-report mismatch that could explain the wrong final classification.
- **Both wrong or incompatible:** compound/ambiguous.
- **Factor reports correct, saved decision correct:** agreement (not independent certification).

Even a clean association is **not evidence of inaccessible internal thinking**. Prompt format changes could explain differences. Four of the factors are also reflected explicitly in the verified machine state, so follow-on ablations withholding redundant machine fields could help distinguish natural-language extraction from direct field usage. Those additional probes are NOT included in the current 64-case proposal.

## Budget proposal — requires separate approval

**One inference-only pod, no training:**
- Candidate: persisted D step-96 Granite 3.3 8B + LoRA, immutable checksum `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`.
- GPU: identical NVIDIA RTX PRO 6000 Blackwell Server Edition, secure cloud, no fallback without equivalent compatibility.
- Maximum GPU rate: **$2.49/hour**.
- Initial recommended total inference cap: **$0.75** (approximately 18 minutes at $2.49/hour; includes bootstrap, cache, 64 factor-only generations and teardown).
- Single-pass generation, deterministic, no sampling; hard `max_new_tokens=128` per case and a wall-clock timeout. Fail closed if price, loading, identity, or spending guards cannot be verified.
- Previous D 64-case state evaluation took 339.7 seconds and cost approximately $0.235; factor-only latency is **not measured**. A $0.75 ceiling is therefore a proposal, **not a guarantee that all 64 prompts will finish**.
- Save results and aggregate analysis immutably under a new, uniquely named diagnostic S3 prefix; verify GPU pod deletion. No sealed-case reuse or production actions.

**Approval has not been granted for this probe.** Previous C/D training or evaluation limits must not be silently reused. No new inference GPU launcher should become executable until a separate user authorization is recorded and verified. The old D/C paid flags remain disabled.

## Independent-data caveat

The proposed 64 cases belong to the *existing frozen development* partition. Factor reporting is elicited independently from the old decision prompt, but **the evaluation cases themselves are not independent held-out data**. This is deliberately a narrow mechanistic development diagnostic, not a new certification pack. It must not be used as training data, accepted as proof of scientific improvement, or used to select/promote a checkpoint.

If the factor-only probe proves informative, a separate genuinely fresh case bank with independent authoring and separate factor + state prompts is required before more expensive scientific claims.

## What has been validated

The CPU preflight tests the exact panel identity, label-exclusion, stable per-prompt hashes, complete/duplicate checks, strict six-field typing, specific provenance errors, and same-case combination with saved D decisions.

**What remains:** separately approved bounded GPU inference, read-back of saved factor generations, full 64-case strict score and paired error decomposition, followed by an honest go/no-go interpretation. No paid experiment or model inference has yet occurred in this PR.
