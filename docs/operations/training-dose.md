# Bounded HF training dose and resume

Future prepared HF jobs pin `epoch-bounded-microbatch-mean.v2`. An accumulation
window ends at the epoch boundary; a short final window divides each microbatch
loss by the actual number of microbatches in that window. The objective remains
the mean of microbatch losses, not a newly claimed token-weighted objective.

For three batches and accumulation two, one epoch consumes `(0,1)` then `(2)`.
It never wraps batch zero into that epoch. Fractional epochs round their batch
budget upward once (`ceil(epoch_fraction * batches_per_epoch)`); at most one
indivisible batch is rounded up. `max_steps` remains an independent optimizer-step
ceiling and the configured scheduler horizon. Resource estimates use the same
window-counting function as execution.

Every checkpoint requires cumulative examples, tokens and batch cursor, including
off-cadence final saves. It retains schedule version, last loss and gradient norm.
Resume verifies the cursor against the frozen schedule and restores the same
optimizer/scheduler/RNG state. An interruption after the last permitted update can
finish as a zero-update resume without rewriting its accepted checkpoint.
Throughput describes the current session; cumulative accounting includes prior
sessions. Old checkpoints lacking this schedule/cursor require explicit replay
with their pinned source, not a silent reinterpretation of their training dose.

The bounded worker currently supports sequential data. A requested shuffle now
fails explicitly instead of being recorded and silently ignored. General shuffled
or streaming corpus training remains a separate capability requirement.

Real tiny CPU regressions check off-cadence counters, three-batch dose, an
independent optimizer reference for partial-window gradients, and final-step
interrupt/resume. These are training correctness checks, not model-quality claims.
