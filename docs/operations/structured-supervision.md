# Prompt/target supervision

New prompt/target records retain `PromptTargetSupervision` with the original
prompt, target and explicit template. A template has exactly one `{prompt}`
followed by one `{target}`; format conversions/specifiers are unsupported. Literal
suffixes after the target are part of the supervised target. Marker-looking text
inside an example does not change its boundary.

Preprocessing keeps each supervised pair indivisible and preserves its rendered
whitespace. If its whitespace-token bound is exceeded, preprocessing rejects the
pair rather than making independent prompt-only or target-only chunks. Plain text
retains its existing chunking policy. The manifest records the new boundary and
chunking semantics.

The worker separately tokenizes the declared prefix and target, masks every
prefix label, and supervises only the target plus required EOS. It checks that
the rendered text and template match the prepared evidence. A pair exceeding the
actual tokenizer context is rejected; silent truncation is forbidden. Legacy
`prompt_target` text-only chunks without structured identity must be reprocessed
from original source. They never fall back to full-text supervision.

The separate original frozen token-ID continuation/R10 exact-token path is not
changed by this contract. A tiny locally initialized CPU model test exercises
the real HF worker using the structured format; it establishes objective wiring,
not useful language-model improvement.
