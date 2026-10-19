# R001 bounded exploratory migration (development experiment)

> Historical initial exploratory policy (8 requests / 3 attempts / 3 seconds).
> Superseded by the [accepted moderated policy](2026-10-19-r001-accepted-policy.md).

## Purpose and evidence status

Test whether the completion-time predictor rejects useful running-request KV migrations.
This is an opt-in exploration policy, not a demonstrated performance improvement.
Results are pending. The formal runner retains independent engine startup; it also
exposes R001 and zero-offpolicy options.

## Policy

`AREAL_KV_REBALANCE_POLICY=exploratory` retains the aggressive positive-gain planner,
then considers a bounded exploratory candidate only if no positive candidate was
selected:

- At most 8 requests per transaction; estimated serialized staging cost at most 1.6
  seconds.
- Source has at least 8 running requests and more than 1.5 times the target's active
  requests.
- Each selected request has predicted remaining execution time at least three times the
  transaction staging estimate.
- Predicted batch-ready/mean-completion and tail regression are each limited to 1.5
  seconds; summed completion-time regression is limited to 0.3 seconds per active
  request.
- At most three exploratory transaction attempts per training run. Stop admitting
  further probes after cumulative observed transaction time reaches 3 seconds. This is
  an admission budget, not a hard timeout on an in-flight transaction.
- Keep memory capacity, matching weight version, exact native KV ownership, request
  lease and training-readiness checks.

The prediction is recorded even when negative. Exploration does not guarantee a speedup
or zero slowdown. A selected plan can still be cancelled if training becomes ready or
ownership changes.

## Development-only tooling

The independent `benchmark/npu_share/run_reuse_dev_bench.py` prepares one workload and
runs fixed 1, shared 1, fixed 2, shared 2. Fixed and shared inference configurations
have separate persistent service pools. Inactive pools are parked; NPU contexts and some
communication allocations can remain. Report these results only as development
validation, never as a formal baseline comparison.

Each run restores a cloned immutable CPU initial-weight snapshot, verifies its digest,
resets request state and random seeds, and creates a fresh trainer. The immutable
snapshot is independent of mutable current-weight DRAM offload backups. Worker identity
and restored weight digests are checked across repetitions. The initial snapshot and
reset machinery are experimental tooling, not R001 research contributions.

Formal comparisons must independently initialize each arm with matched warmup and timing
rules through the unchanged `run_bench.py`.

## Required observations

Record actual successful migration count, transferred KV bytes, transaction cost,
prediction error and six-batch training-loop time. Compare the first and second pairs
separately. A runtime difference without actual migration is not evidence of migration
benefit.

## Historical snapshot replay

On 15 saved snapshots, the aggressive planner selected 2 requests in one snapshot.
Exploratory mode selected 43 requests across 9 snapshots (8 bounded probes and one
positive-gain plan); 5 snapshots retained admission priority and one had no acceptable
plan. These are independent offline candidates, not executed migrations; a live run also
enforces the per-run exploration budget and ownership revalidation.

## Tool validation issues

The first session failed before any training because vLLM-Ascend native wake-up rejects
FRACTAL_NZ weights. The independent development launcher now sets `weight_nz_mode=0` for
both arms and validates a park/reset cycle immediately after each service starts. A
subsequent CLI serialization failure was corrected by JSON-encoding `additional_config`.
Both failed session workspaces are retained; neither provides migration performance
evidence. This changes the development environment only, not the formal benchmark
defaults.

## Live handoff failure and lifecycle correction

The session `20261019-115702-0700_math_fsdp_c16_b6_6427a3` completed fixed repetition 1
in 193.365 seconds (576 answers, 137204 generated tokens). Shared repetition 1 failed
before completing the comparison. Seven requests exported and transferred KV, but zero
target imports were confirmed. The transaction took 6.861 seconds: 1.695 seconds staging
and approximately five seconds waiting for imports. These seven requests must not be
counted as successful migrations.

Training preparation paused the target before it could resume the imported requests.
Admission retries then exhausted. The controller now fences training borrowing, shared
drain, fixed publication, generic pause and teardown against outstanding handoff
completion. Async pause waits through a worker thread to keep the event loop responsive.
The relevant 85 regression tests and formatting checks passed. The next development
session was launched after these corrections; performance results remain pending.

A preceding development-only failure came from checkpoint paths differing between
trainer and persistent-service mounts. The reuse worker now translates the scoped
session checkpoint path before loading; the formal runner remains unchanged.
