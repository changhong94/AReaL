[中文](M2-readiness-aware-scheduling.zh.md) | [Overview](../core-mechanisms.md)

# M2: Request scheduling for training readiness

## Problem and mechanism

Predict remaining work and choose draining, queued relocation or running migration to
make training ready sooner.

Up to three completed batches of statistics → residual-time prediction → data/device
readiness objective → cost-aware scheduling → measured-cost feedback.

## Traceable implementation records

- [R004](../optimizations/R004-adaptive-draining.md)
- [R019](../optimizations/R019-tail-estimation-and-sibling-ablation.md)
- [R001](../optimizations/R001-runtime-request-rebalancing.md)
- [R005](../optimizations/R005-queued-request-rebalancing.md)
- [R012](../optimizations/R012-request-leases-and-migration.md)
- [R013](../optimizations/R013-question-affinity-and-concurrency.md)

These are components of one mechanism, not independent novelty claims. Stable R
identifiers and historical evidence remain unchanged.

## Limitations and negative results

R019 sibling scaling was removed; R013 affinity has no established isolated benefit;
R005 found no eligible moves in inspected runs; the R001 timing gate fired zero times
and remains default-off; R020 TP2 tail tests are negative evidence.

## Required isolated validation

Compare migration on/off within the same sharing, memory and publication configuration;
repeat runs and report tokens, confirmed imports, transaction cost, training wait and
quality. Separate prediction error from scheduling benefit.

Overall shared-system speedup does not establish the isolated benefit or novelty of this
mechanism. See the
[accepted-policy evidence](../experiments/2026-10-19-r001-accepted-policy.md).

## Proposed extension

[R025: Capacity-aware adaptive inference concurrency](../optimizations/R025-adaptive-inference-concurrency.md)
adds a research proposal for dynamic admission targets. It is not implemented or
validated.

[R026: larger-TP live-tail migration](../optimizations/R026-tail-acceleration-via-tp-migration.md)
is a new proposal. It first requires evidence that a prewarmed larger-TP engine reduces
remaining decode time enough to pay for cross-TP KV redistribution. R020 remains a
negative small-model control.
