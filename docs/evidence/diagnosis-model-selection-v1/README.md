# Diagnosis model selection v1

This evidence record freezes the comparison used to select the next Hephaestus diagnosis-role fine-tuning candidate.

## Selection

`mistralai/Ministral-3-3B-Reasoning-2512` at immutable revision
`4a36357c811bf511a7b625d132e12f22408aac91` is selected as the
**candidate for diagnosis role-specific fine-tuning**.

This is a candidate-selection decision only. It is **not** production
certification, promotion, or lineage mutation.

## Evidence lineage

- Corrected diagnosis foundation bakeoff: `diagnosis-foundation-v1-35515098798`
- Paired micro-LoRA adaptability experiment: `diagnosis-adaptability-v1-35525671435`
- Frozen diagnosis pack SHA-256:
  `b1a1909c2fd89640a7c1981c5c9de8f951bf1083a8a76c5be653456ca42fcf8c`
- Selected Ministral adapter SHA-256:
  `15f29e4a12b912a8c4b917833676b5eefb40017cc6ace3467fe2a51beed7cd8e`

## Matched intervention

Both finalists received the same 72-case diagnosis curriculum, rank-8 /
alpha-16 LoRA, 32 optimizer steps, learning rate 5e-5, seed 11, and
98,304 padded training-token slots. Adaptability was measured on the same
48 untouched held-out adversarial cases before and after LoRA.

| Metric | Granite 4.2 3B | Ministral 3 3B Reasoning |
| --- | ---: | ---: |
| held-out quality before | 70.2083 | 88.9583 |
| held-out quality after | 73.9583 | 99.4792 |
| quality gain | +3.7500 | +10.5208 |
| gain / 1k supervised tokens | 0.2926 | 0.7825 |
| post schema compliance | 77.08% | 100% |
| post evidence grounding | 75.00% | 96.53% |
| post hallucination rate | 25.00% | 3.47% |
| post exact-contract pass | 54.17% | 89.58% |
| post confidence calibration | 58.33% | 100% |
| post mean latency | 22.7500 s | 3.4808 s |
| mastered-capability regressions | D3 -50.625 | none |

Ministral retained D2 and D3 at 100, improved D4/D5/D6 to 100, and
improved D1 to 96.875. Granite improved several weak domains but incurred
a severe D3 regression and degraded schema/grounding reliability.

## Frozen disposition

Ministral is the selected **foundation candidate** for the next
diagnosis-specific fine-tuning stage. Future work must preserve an
independent certification set and must not reinterpret this selection
record as production certification.
