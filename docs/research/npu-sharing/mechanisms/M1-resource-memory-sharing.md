[中文](M1-resource-memory-sharing.zh.md) | [Overview](../core-mechanisms.md)

# M1: Training-priority resource and memory sharing

## Problem and mechanism

When to borrow devices, how to handle memory pressure, and how to return capacity safely
to training.

Role switching and training-priority admission → dynamic physical KV pages → partial
offload → training-state prefetch.

## Traceable implementation records

- [R007](../optimizations/R007-single-npu-and-borrowing.md)
- [R002](../optimizations/R002-training-priority-concurrency.md)
- [R006](../optimizations/R006-dynamic-kv-pages.md)
- [R017](../optimizations/R017-pressure-driven-partial-offload.md)
- [R014](../optimizations/R014-optimizer-prewarm-and-pure-status.md)
- [R015](../optimizations/R015-training-cache-parking-prefetch.md)
- [R018](../optimizations/R018-retained-pause.md)

These are components of one mechanism, not independent novelty claims. Stable R
identifiers and historical evidence remain unchanged.

## Limitations and negative results

R018 retained-KV pause remains a non-default alternative. R015 mechanism validation does
not establish strict numerical equivalence or stable speedup. Report memory savings
separately from end-to-end latency.

## Required isolated validation

Ablate dynamic mapping, partial offload and prefetch separately under matched requests
and tracing; vary memory pressure and measure physical memory, transfer bytes, switching
time and training quality.

Overall shared-system speedup does not establish the isolated benefit or novelty of this
mechanism. See the
[accepted-policy evidence](../experiments/2026-10-19-r001-accepted-policy.md).
