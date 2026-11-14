# Coverage probe v2 — independently verified measurement repair (2026-11-14)

## Decision

**Repaired the actual source-level measurement blind spot.** The legacy
`analyze_dataset_task_coverage` only counted separate JSONL `prompt` and
`target` keys and missed the original approved preprocessing wrapper
`<|prompt|>\n{prompt}\n<|target|>\n{target}` embedded in the record's `text`.

The v2 probe detects complete **nonempty, unambiguous** embedded wrappers,
while reporting marker-only rows separately and checking parseable JSON-object
targets separately. A row with both explicit and embedded representations is
counted once. Empty/malformed/unordered/duplicated wrappers do not establish
supervised instruction coverage. It retains the frozen evaluation pack and
source bytes unchanged and does not rewrite any prior reports.

## Actual authenticated source acceptance

[GitHub Actions read-only S3+full suite #43639998701](https://github.com/kys928/Hephaestus/actions/runs/43639998701) fetched and SHA-256 rehashed the original historical 54,878-row dataset
`f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
and the later controlled 171,295-row dataset
`bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`
from the original Czech regional S3. The pinned source checkout ran the
actual modified v2 probe against those original bytes using the unchanged
`semantic_behavior_v1` evaluation pack.

| Verified measurement | Original | Later dataset |
| --- | ---: | ---: |
| Rows | 54,878 | 171,295 |
| Separate prompt/target pairs | 0 | 0 |
| Literal embedded marker pair rows | 0 | 35,774 |
| Complete nonempty embedded prompt/target examples | 0 | **35,568** |
| Empty or malformed embedded pairs | 0 | 206 |
| Parseable structured JSON target examples | 0 | **0** |
| Combined v2 prompt/target coverage count | 0 | **35,568** |
| Supervised instruction structure present | no | **yes** |
| Supervised structured JSON examples present | no | **no** |
| Overall frozen task coverage | incomplete | **still incomplete** |

Thus the original diagnosis was supported on the baseline, and the candidate
has a genuine material instruction-format coverage improvement, but the
candidate does **not** satisfy the frozen structured-target evidence
requirement. Neither the positive signal nor the existing `regressed` frozen
hard-gate outcome proves semantic competence or repair efficacy.

## What was changed

- `src/hephaestus/diagnosis/probes.py`: shared v2 format-aware counting.
- `tests/test_targeted_post_failure_probes.py`: nonempty wrapped strings,
  valid and malformed JSON, empty/inverted/repeated markers, streaming
  iterables, and deduplication across both representations.
- `tests/test_prometheus_authenticated_contrast.py`: the previous fixture
  that asserted the *known bug* is updated to assert the corrected behavior
  without rewriting historical original data or reports.

The original evidence in `docs/evidence` stays immutable and historically
true of the v1 implementation. The v2 output advertises
`coverage_probe_version=2` so consumers can detect measurement semantics.

## Next blockers — distinct gates

1. **Domain capability engineering:** genuine structured JSON target coverage
   is still unverified/absent in the later source; the frozen hard evaluation
   also still reports regression. Fixing a *probe* does not authorize a data
   recipe, evaluator, Granite or checkpoint repair. Any changed dataset and
   new training/evaluation need their own explicit scientific approvals.
2. **Diagnostician reliability research:** now use these independently
   rehashed same-lineage source states and precise positive/missing
   predicate labels in an initial **retrospective, bounded frozen-role pilot**.
   The eight-slot original-Granite design is not required for such a
   narrower diagnostic-instrument test. Preserve source materiality,
   repeated seeded inference and abstention.
3. **Original six-state Granite autonomy/certification:** this different
   training lineage cannot provide independently matched, original-Granite
   positive eval-integrity observations or prove parent repair/generalization.
   Genuinely new independently labeled original-Granite evidence is required
   **only before making that particular stronger claim**; not before fixing
   this parser or studying another real source-linked diagnostic case.

No model weights, final holdouts, source datasets or historical scorecards
were changed. No Planner, Granite repair, GPU training, promotion or
production routing was authorized.
