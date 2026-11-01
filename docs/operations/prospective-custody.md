# Prospective custody and CPU rehearsal

The custodian-side `ProspectiveCustodyRunner` connects model/checkpoint/job
identities, complete three-seed development selection, exposure exclusion,
balanced canonical holdout commitments, actual CPU generation, raw token
receipts, strict scoring and paired analysis. It verifies Ed25519 grants against
public keys configured out of band. It has no production signing or independent
certification implementation.

The training operator must not operate the real custodian service or possess
its holdout, signing key or writable publication ledger. Host-local locks and
append-only state prevent ordinary concurrent/restarted invocations from
reusing an exposure session. They do not defend against an administrator
deleting or cloning custody state. Cross-host execution requires a shared
fenced ownership service; it is unsupported here.

One grant binds the design, execution source/framework profile, all model and
tokenizer identities, every development attempt and selection, the exposure
ledger and final holdout commitment. The session is consumed durably before
generation. A completed identical submission returns cached evidence. A failed
or unknown session remains consumed, including after process reconstruction;
there is no automatic fresh holdout consultation. Partial completed generation
artifacts remain referenced. Recovery needs separate custodian reconciliation,
not another training attempt on the same final items.

Candidates must use the baseline's exact base and tokenizer. Each adapter must
be the model component of its retained training checkpoint, bound to its seed
and actual prepared job. Confirmatory mode additionally checks original model
and tokenizer file hashes, the original 96 training rows, chat/EOS supervision,
one full 96-update recipe and original development bytes. Missing originals
fail closed. A signed commitment cannot prove unbiased sampling: the real
custodian must retain and privately reproduce its sampling seed/draws.

Raw receipts retain input and generated token IDs, full decoding, EOS and
truncation. Scoring removes only the terminal EOS token. The publication bundle
contains primary and all completed seed analyses, with Bonferroni-adjusted
secondary tests. Rehearsal flags always prohibit confirmatory effect claims,
external certification and promotion. The v2 protocol is still a preparation
draft awaiting independent pilot acceptance; infrastructure does not freeze it.

Cold verification requires a separately retained publication content identity,
the entrusted public-key mapping and the artifact store. It rechecks signed
scope, development scores, model identities, partition/support, exposure,
pairing, EOS flags and statistics. With original model bindings it also
verifies bytes, retokenizes and decodes receipts; optional actual model
reexecution compares every original token. Use the pinned source/framework
profile for replay. This is software verification, not independent human
scientific adjudication.

## Bounded authentic rehearsal

Install the existing `test`, `peft-training` and `independent-certification`
extras. From the checkout, with no network or CUDA:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m hephaestus.development.prospective_rehearsal run /absolute/new-run
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m hephaestus.development.prospective_rehearsal cold /absolute/new-run
```

The root must be new. The helper creates a real 37,472-parameter randomly
initialized GPT-2 model and byte tokenizer, then runs seeds 1117/2227/3337 for
four genuine LoRA optimizer updates each. Eight public synthetic final examples
exclude the historical 144 exposed examples. Its newly generated fixture key
is labeled `fixture-rehearsal-only-not-independent`; it supplies no authority
for a confirmatory study. Raw data, weights, state and outputs stay outside Git.

The campaign rehearsal returned baseline 0/8 and selected candidate 0/8 exact
correct, gain 0, McNemar p=1 and a degenerate observed bootstrap interval [0,0].
All eight candidate outputs lacked EOS and hit the 16-token cap. Every seed's
development result was tied; seed 1117 won the predeclared tie-break. This tiny
observed interval does not establish population equivalence. No behavioral gain
or seed reliability was demonstrated. Separate-process cold model reexecution
reproduced the outputs. `test_prospective_custody_pipeline.py` also covers
authority/scope substitution, complete attempt selection, interrupted sessions,
immutable publication roots, forged termination and checkpoint substitution.

Original SmolLM2 base/tokenizer bytes, the separate private R10 archive,
independent custody and blinded power/EOS pilot acceptance remain unavailable.
Issue #152 stays open. No paid experiment, promotion or external certificate
is authorized by this rehearsal.
