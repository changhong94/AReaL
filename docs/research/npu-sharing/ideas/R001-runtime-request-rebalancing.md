English | [中文](R001-runtime-request-rebalancing.zh.md)

# R001: Cost-aware rebalancing of already-running requests

Created: 2026-10-15 (America/Vancouver). Status: proposed; no implementation, causal
performance result or novelty claim.

## Motivation and evidence

[E001](../experiments/20261015-144720-megatron-c32-b6.md) shows that the shared
training/inference card produced 21.3% of generated tokens. All five post-recovery
queue-stealing decisions returned `no_queued_work`. Current R005 deliberately excludes
running requests. A source engine can therefore have substantial work while no requests
qualify for rebalancing.

## Hypothesis

Moving selected long-running requests from the dedicated inference engine to the shared
engine can shorten the critical batch completion time when the predicted remaining work
exceeds migration, reconstruction and compute-contention costs. Equal token counts are
not the objective; total RL loop time and correctness are.

## Candidate decision rule

For candidate set S, estimate:

`gain(S) = predicted critical completion without S - predicted critical completion with S`

The second estimate must include source interruption, transfer or prompt/KV replay,
remaining destination decode, effects on other requests and training slowdown. Only act
when the predicted gain exceeds an uncertainty margin and memory/version constraints
hold. Estimate service rates and remaining lengths from the existing rolling window (up
to three completed batches, fewer during warm-up), without using future observations.
The exact estimator and fallback remain design work.

## Correctness constraints

- Reserve training memory first and recheck destination headroom at admission.
- Preserve request identity, generated prefix, sampling/termination semantics and
  compatible weight versions; do not silently resume an old prefix under new weights.
- Define whether migration transfers KV or reconstructs it. Do not assume KV transfer
  exists or is free; record the actual path and its measured cost.
- Guarantee at-most-one active owner, no lost answers and bounded retries.
- Add hysteresis/cooldown to prevent repeated back-and-forth migration.
- Fail closed when cost estimates, version compatibility or memory checks are invalid.

## Implementation and validation plan

1. Log candidate snapshots and counterfactual estimates without moving requests.
1. Validate handoff ownership, prefix/version integrity and failure recovery on a small
   run.
1. Add an optional policy, preserving the existing queued-only baseline.
1. Compare fixed allocation, shared queued-only, and shared queued-plus-running
   policies.
1. Use identical model/data/configuration/logging and paired repeated seeds. Use replay
   when isolating execution cost; live sampling may generate different answers.
1. Measure loop/batch time, throughput, TTFT/TPOT/latency, training slowdown, migrated
   tokens/bytes, replay cost, forecast error, memory pressure and reward/quality.
1. Include short/long tails, insufficient memory, imminent publication, and cases where
   doing nothing is better. Report regressions and variance, not only best runs.

## Related work and outcome

Related-work review: pending. Implementation commit: none. Results: none. Update this
section with experiment IDs, negative findings and revised hypotheses.

## Proposed implementation locations and paper framing

Expected touchpoints: `areal/infra/queue_rebalance.py` (candidates/cost model),
`request_router.py` (leases/reservations), `areal/engine/vllm_ext/queue_rebalance.py`
(safe engine interruption), and `rollout_controller.py` (versions/recovery). These are
proposed locations, not completed code.

Existing migration normally replays the prefix rather than transferring KV. R001 must
choose a compatible destination version and account for replay; reversing the existing
route does not automatically preserve one policy version. Ownership, partial output,
cancellation acknowledgement, model versions and training contention interact.

The candidate research question is whether future-training-aware critical-path benefit
can reliably exceed all migration costs. Novelty over generic work stealing and
inference migration requires literature comparison. No R001 speedup exists; E001’s 9.37%
belongs to the prior combined strategy.

See [R005 queued rebalancing](../optimizations/R005-queued-request-rebalancing.md),
[R012 continuation](../optimizations/R012-request-leases-and-migration.md), and
[R019 prediction limits](../optimizations/R019-tail-estimation-and-sibling-ablation.md).
