English | [中文](2026-10-20-incremental-budget-timing.zh.md)

# Incremental memory budget and handoff timing validation

Date: 2026-10-20 (America/Vancouver). Development-only warm-engine comparison; not
independent-start formal evidence.

Two fixed/shared pairs used FSDP, FULL graph inference, two NPUs, 96 math questions, six
answers each, concurrency 16, six batches and off-policy 2. R001 KV rebalance and R025
capacity admission were enabled for shared. R027 is a proposal and was not enabled.
[Machine-readable evidence](2026-10-20-incremental-budget-timing.json).

| Pair | Fixed seconds | Shared seconds | Shared slowdown | Fixed/shared reported tokens |
| ---- | ------------: | -------------: | --------------: | ---------------------------- |
| 1    |         75.50 |          99.31 |          31.54% | 133017 / 137940              |
| 2    |         75.24 |          99.63 |          32.41% | 134271 / 133559              |

## Verified corrections

FSDP now reports only exactly matched, mapped, wholly inactive private workspace
segments on the current stream. The policy credits no more than the gradient
reservation; configured training and inference workspaces, safety, future optimizer and
publication reserves remain. Unknown backends retain zero credit; Megatron ordinary
allocations are not credited. Both shared runs admitted concurrent inference in all six
batches. Later batches credited approximately 3.74 GB, rather than treating the whole 11
GB resident workspace as available.

Publication resets its timing origin every cycle, even without a drain; consumption
clears the timestamp. Measured handoffs were 2.10–2.86 seconds rather than stale
multi-batch cumulative values. Targeted tests: 78 passed; Ruff checks passed.

## Performance interpretation

These corrections did not establish end-to-end acceleration. Waiting for the preceding
background tail took 20.16 and 19.98 seconds. Shared device inference step-window unions
were 24.54 and 23.18 seconds; these are host/device-trace windows, not measured compute
utilization. Several batches already had generated answers while training awaited
publication/return. This motivates R027's critical-path profitability decision instead
of maximizing inference occupancy.

Token counts differ, so these are workload-matched RL comparisons, not identical
generated-output replays. Do not subtract an earlier independently started run from
these warm-session measurements to claim a causal speedup. No formal independent-start
confirmation was run because both development pairs regressed. Initial aborted
initialization is retained separately; it contributed no training result.
