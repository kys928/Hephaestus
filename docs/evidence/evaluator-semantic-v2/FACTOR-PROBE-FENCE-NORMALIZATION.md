# Factor-only inference: strict JSON versus fenced JSON

The paid probe completed 64/64 reports but **0/64** passed the strict raw-JSON contract. An independent CPU-only structural audit [#38963204051](https://github.com/kys928/Hephaestus/actions/runs/38963204051) determined:

- All 64 began with a Markdown code fence.
- All 64 included a complete JSON object with **exactly six factor keys**.
- None reached the token limit or inference deadline.
- The old seven-field Evaluator schema was not emitted.

The original score of 0/64 remains authoritative for *strict formatting*. A distinct exploratory CPU-only scorer will accept **only complete JSON objects surrounded by exactly one Markdown JSON fence**, without changing any field or using expected labels for repair. That scorer will test whether the six values match independent frozen reference evidence, then pair results with saved D decisions.

No new GPU inference or changes to checkpoints or scientific gates are authorized by this audit.
## Exact-wrapper normalization outcome

The first conservative normalizer could not validate a complete opening-and-closing Markdown wrapper on any of 64 responses. Original strict score stays 0/64. An additional read-only delimiter audit will characterize precisely what precedes and follows the embedded JSON object before any exploratory factor-value scoring; no GPU rerun authorized.

## Actual saved wrapper identified

Independent delimiter audit [#38963653286](https://github.com/kys928/Hephaestus/actions/runs/38963653286) confirmed the exact structure was a Markdown JSON fence containing **one JSON array with one object**, not the previously expected directly fenced object. This explains why the conservative object-only normalizer accepted 0/64. A second, explicitly exploratory normalization accepts only this *complete singleton-array* envelope with all six typed keys. The original strict-failure result stays unchanged; the next CPU audit will determine factor accuracy.

## Exploratory scientific signal

A complete, exact singleton-array Markdown wrapper was normalized on 64/64 original raw completions; the original strict-format result remains 0/64. Factor truth recovery after structural normalization was 23/64 all-six correct, with provenance 64/64, confirmation 64/64, completeness 63/64, gates 61/64, variance 45/64, and effect 28/64. The enhanced CPU rescore will now also report factor-value and per-scientific-state confusion so this result can be interpreted without overclaiming internal reasoning.
