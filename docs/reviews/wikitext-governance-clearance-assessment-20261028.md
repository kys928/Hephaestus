# WikiText exact-subset governance assessment — 2026-10-28

Status: **HOLD**. This is an implementer's evidence assessment and a concrete
external-review handoff, not an independent rights, privacy or scientific
clearance. No protected reference was accessed; no corpus was downloaded by
this investigation; no person was contacted or appointed custodian.

## Exact object under review

The candidate is `Salesforce/wikitext` at
`b08601e04326c79dfdd32d625aee71d232d685c3`, file
`wikitext-2-raw-v1/train-00000-of-00001.parquet`, SHA-256
`e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
The existing technical screen selected 7,232 rows / 13,380 continuation windows,
with selection commitment
`sha256:87ae1f4ab0f71693f0f20e733ebd7aa8ff25edc252f51f9ea306ef74ac7fe69e`.
These are inherited measurements from PR #114, not measurements rerun here.
Any privacy/leakage exclusion that changes the selection needs a new commitment,
new counts and a new externally accepted review; preserving 13,380 at the
expense of excluding a contaminated window is not a valid goal.

## Authoritative source checks

The following public pages were checked on 2026-10-28. URLs identify the sources;
web pages can change, and this note does not claim archived immutable custody.

| Source | Observation | Consequence |
| --- | --- | --- |
| [Pinned publisher card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) | Metadata lists CC BY-SA 3.0 and GFDL, while the licensing section names CC BY-SA 4.0; source-collection and personal-information sections provide no substantive detail. | The discrepancy is in the exact pinned card itself. A current badge does not resolve it. |
| [Dataset author's maintained mirror](https://state.smerity.com/direct/smerity/state/01HRTBE0WN2QV8PRC6JDH22DGW) | Stephen Merity mirrors the original release description and supplies working download links and citation credit. It names Creative Commons Attribution-ShareAlike without a version. | This improves provenance and supplies an author-maintained resource, but does not select 3.0 versus 4.0 or provide a row-to-article revision map. Do not substitute this archive for the pinned Parquet file. |
| [CC compatibility rules](https://creativecommons.org/compatible-licenses/) | Contributions to adaptations of BY-SA 3.0 material may use that version or later BY-SA versions. | This is a possible compatibility route for adaptations, not evidence that this exact corpus was relicensed or that all original obligations were satisfied. |
| [Wikimedia Legal's 2023 change explanation](https://meta.wikimedia.org/wiki/Wikimedia_Foundation/Legal/2023_ToU_updates/About#Section_7_Licensing_of_Content) | The explanation describes future contributions switching from 3.0 to 4.0 and articles becoming 4.0 as updated. | Inference: a current Wikipedia license cannot, by itself, establish the license/attribution chain of historical extracted bytes. |
| [Wikimedia reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content) | Reuse must follow the applicable license; attribution may use article URLs or equivalent author credit. Imported content can have additional requirements and may be CC-only. | A blanket GFDL choice or dataset citation alone is insufficient evidence for article-specific obligations. |

The apparent original [Salesforce blog URL](https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset)
redirected to the AI category page on opening. The author's link to
`blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/`
returned 404. Search snippets preserving old prose must not be treated as a
currently available original license instrument.

No conclusion about whether model weights are adaptations, or what license
would apply to distributed weights, is established here. The authorized rights
review must state its intended-use assumptions explicitly.

## Concrete rights and privacy review handoff

An authorized independent source reviewer needs to return a decision bound to
the exact candidate, file SHA, accepted processing profile and row/window
commitment, with evidence references for:

1. Applicable historical text license(s) and the reconciliation of the pinned
   card's competing statements; distinguish source-copying, internal training,
   processed-corpus distribution and model distribution.
2. An attribution/provenance mechanism accepted for the selected material:
   article or revision mapping, author/history links where appropriate,
   dataset authors/citation, license notices, modification notices, and any
   imported-content exceptions. The present card and Parquet `text` field are
   not a verified article/revision manifest.
3. Privacy review of the selected rows and the 54 excluded regex candidates,
   with documented treatment of sensitive or identifying content that regexes
   miss. Flag exclusion is measured; exhaustive absence of PII is not proven.
4. Disposition of the three earlier retrieved lexical near-duplicate pairs,
   an appropriate complete source/reference comparison method, and semantic
   contamination review. The earlier heuristic candidate retrieval is not an
   exhaustive clearance.

The 54 regex-candidate exclusions do not need to be reintroduced to conduct
review. Review original text inside the authorized review environment, and
return aggregate decisions rather than publishing source or protected text.

## Independent reference custody: inputs actually missing

`configs/data/quarantine_source_screening_v1.json` explicitly sets training and
protected-reference access to false. The existing provisioning request in
`scripts/run_quarantine_source_screening.py` has empty reference refs and null
custodian/key/protected-scope fields. This is deliberate absence of authority,
not an integration value that the implementer may fill with generated data.

Known incomplete metadata are already recorded there:

- Original training bytes digest
  `sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
  and storage key
  `hephaestus/scientific/v1/runtime_bindings/first-bounded-scientific-training-001-33866198758/dataset/trainable.jsonl`.
- Public diagnostic probe file
  `docs/evidence/first-targeted-post-failure-diagnostics-001-33874832481/probe.json`,
  digest `sha256:c943334f2f6e7162380be74748506d26dd13542fece21d42672d35f3abbfc2e9`.
- Public development behavior pack; this exposed pack is not a sealed holdout.
- Independent holdout: no accepted identity is supplied.

These facts do not establish four complete exposure inventories. This review
did not read the original training or protected data to search for a custodian.

| Required external input | Existing acceptance boundary |
| --- | --- |
| Actual custodian ID plus independently pinned 32-byte Ed25519 public key and accepted statement digest | `verify_custodial_reference_bundle` in `src/hephaestus/policy/custodial_references.py` |
| Four distinct training/development/diagnostic/holdout manifests, each binding original-text content ref, positive row count, sealing time, normalization and complete exposure inventory | `CustodialReferenceManifest`; no test fixture or freshly generated replacement can prove historical custody |
| Separately authorized protected comparison scope binding exact source hashes and all four manifests; no training; execution inside custodian environment | `custodial-contamination-audit-scope.v1` and signed `custodial-reference-access.v1` |
| Full-corpus PII, exact/near-duplicate, semantic contamination and frozen-evaluation leakage audit receipts, accepted source-rights evidence and exact processing profile | `IndependentDatasetReviewPolicy.require_clearance` in `src/hephaestus/policy/dataset_review.py` |
| Native selection and acquisition authorization tied to the actual selected candidate/decision | Original staged approval gates, retained independently of technical materializer tests |

The two access-scope schema families serve different boundaries: the custodial
signature verifier accepts the protected custodian scope, while dataset
admission expects a separately trusted `DatasetScreeningAccessScope`. One
must not simply replace the other. Both need consistent source/reference
identities and real external acceptance.

## Work that can proceed and the real stop condition

The exact token-ID materializer, its adversarial tests, durable-state recovery
verification and PR dependency checks can proceed now. Source-only technical
validation does not require representing its output as admitted training data.

A genuine new campaign starts at 0/45 and may advance only as its actual native
inputs and approvals permit. GPU spending permission supplies compute authority;
it does not supply absent reference texts, a custodian signature, source-rights
facts or independent scientific certification. Changing sources can simplify
license provenance, but cannot substitute for the missing frozen-reference
custody and contamination audit.
