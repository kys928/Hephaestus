# Analytic power checks for the unexecuted #152 design

These calculations do not accept a design or access protected examples. They
supplement the simulation in [protocol v2](152-prospective-protocol-v2.md).
`scripts/analyze_prospective_power.py` reproduces the fixed sensitivity grid using
`evaluation.pilot_power`; no model, data sampler or custody grant is involved.

## Model and derivation

Assume independent identically distributed canonical paired outcomes with win
probability `w`, loss probability `l` and tie probability `1-w-l`. Then discordant
count `D` is binomial `(N,w+l)` and wins conditional on `D=d` are binomial
`(d,w/(w+l))`. Sum their joint probability over integer `(d,wins)` satisfying all
three conditions: positive exact two-sided McNemar p < alpha, observed net gain
at least 5 percentage points, and losses at most `floor(.01*N)`. This finite sum
has no Monte Carlo error. It is evaluated numerically with a normalized PMF
recurrence; omitted marginal probability is reported. Floating roundoff is not
certified by interval arithmetic.

This is **only an IID upper bound for a subset of the gates**. Family, format,
bootstrap and termination requirements can reduce full power. Fixed family
quotas with different probabilities, proposal-weighted sampling without
replacement, semantic dependence and shared templates do not obey this IID
model automatically. Multiplying by an EOS probability additionally requires
independence between termination and the other outcomes. Do not describe this
table as the exact power of the actual stratified experiment.

| N | True gain | True loss | Subset power, alpha .05 |
| --- | --- | --- | --- |
| 256 | 5pp | 0.5% | 47.83% |
| 512 | 5pp | 0.5% | 48.86% |
| 2,048 | 5pp | 0.5% | 49.17% |
| 256 | 8pp | 0.5% | 83.87% |
| 512 | 8pp | 0.5% | 94.89% |
| 1,024 | 8pp | 0.5% | 98.40% |
| 2,048 | 8pp | 0.5% | 99.80% |
| 512 | 8pp | 1.0% | 59.29% |
| 2,048 | 8pp | 1.0% | 51.63% |

A true effect at the observed 5pp acceptance boundary has approximately half
power even as N increases. A true loss rate exactly at its hard observed limit
has a similar problem, with discrete floor effects. Increasing N alone cannot
turn either boundary into a reliable 80%-power claim. The earlier v2 simulated
50.6% estimate is compatible with this calculation's 49.17%; it was not evidence
of adequate power at 5pp.

## Termination, pilot precision and seed claims

If each output independently terminates with probability .999, the all-EOS gate
has probability `.999**N`: 77.40% at N=256, 59.91% at N=512 and 12.89% at
N=2,048. This gate can dominate feasibility. These are sensitivity assumptions,
not estimated rates or permission to relax a gate after exposure.

Even zero losses among 64 independent pilot pairs gives a two-sided exact 95%
binomial loss interval of `[0, .05601]`, not proof that losses are below 1%.
Three successes in three IID randomly sampled training seeds would give an exact
95% lower reliability bound of only .29240. The predeclared fixed seed values
1117/2227/3337 do not themselves justify a random-seed population model.

The development-selected seed remains one primary comparison. Three separate
committed-seed comparisons require their prospectively specified multiplicity
procedure; all three successful observed seeds still do not establish universal
seed reliability. The script also reports alpha `.05/3` sensitivities.

Choose the smallest worthwhile design only after the separate public pilot and
an explicit decision about target effect, termination and sampling assumptions.
If the recipe is not feasible, retain a no-launch result and revise prospectively;
do not enlarge N or soften thresholds just to produce a positive conclusion.

## Verification and methodological sources

Six tests compare the finite calculation with complete small multinomial
enumeration using the existing exact scorer, exercise the 5pp boundary, and
check closed-form boundary intervals and PMF normalization. No generated
scientific evidence is committed to Git.

Primary references: [exact McNemar binomial test](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
[binomial PMF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html),
[exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
and [paired bootstrap resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
