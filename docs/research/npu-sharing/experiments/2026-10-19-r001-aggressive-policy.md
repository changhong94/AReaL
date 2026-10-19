# R001 aggressive KV scheduling experiment

Status: both paired NPU runs completed successfully. No live KV migrations occurred; no
migration speedup has been demonstrated.

## Policy

This experimental policy is selected with `AREAL_KV_REBALANCE_POLICY=aggressive`
together with `AREAL_RUNNING_REBALANCE=kv`. The default policy remains `conservative`.
Set the variable inside the training container; a host variable alone is insufficient.

The primary objective is the estimated time to complete enough fully accounted question
groups for the next training batch. A question requires all its answers, across live and
completed requests. When the visible horizon is incomplete, the explicitly logged
fallback minimizes mean request completion time. Both objectives require no predicted
increase in summed completion times and limit predicted final-tail regression to
`min(0.2 s, 2% of remaining tail)`.

For m selected requests, all moved requests wait for a common staging barrier
`coordination + sum(stage_i)`. This matches the existing protocol:
export/transfer/register each request, then abort the source set together. Unmoved
requests pay estimated per-device worker stalls instead of the full HTTP transaction
duration. Snapshot cost is not multiplied by m; measured planning cost is deducted once.
A 5 ms net-gain margin replaces the conservative 600 ms initial margin.

Initial stage and coordination estimates are 180 ms/request and 80 ms/transaction,
respectively. Initial source and destination stalls are 20 and 50 ms/request. These are
hypotheses informed by the previous 273 ms one-request handoff, not validated
multi-request speed measurements. Successful transactions update rolling medians over at
most 32 observations. Operation-wide timers include source synchronization, destination
allocation, and import/copy completion. Source and target HCCL times are not added
together.

If a destination concurrency bucket is missing, the aggressive estimator may borrow the
source's measured concurrency slope, normalized by destination speed. The resulting
destination speedup is capped at 20%. This extrapolation is explicitly logged. It does
not assert inverse-concurrency scaling.

The simulator adds unmoved-device stalls after decode simulation; it does not fully
resimulate concurrency changes during those stalls. Remaining lengths are conditional
mean estimates. These approximations require comparison against measured training
completion, tokens, and handoff outcomes; positive predictions do not prove speedup.

## Unchanged safety requirements

Same weight version, exact computed prefix, atomic lease reservation, physical staging
headroom, destination KV capacity, one-use import ticket, and training-readiness
rechecks remain mandatory. The policy still prioritizes ordinary queued admission and
makes at most one bounded migration transaction per recovery (up to 16 requests).

## Validation

67 related CPU tests passed. Selected pre-commit hooks and `git diff --check` passed.
Tests cover group completeness, common staging barrier, bounded tail regression,
flat-service no-op, version/budget gates, and existing KV handoff and rollback behavior.

Paired shared-two-card runs use arithmetic RL, FSDP, 96 questions, 6 answers/question,
concurrency 16, six batches, and identical tracing. Both arms run the same source with
only the policy environment variable changed. No forced migration override is used.

## Measured results

| Policy       | Training loop | Answers | Generated tokens | KV migrations |
| ------------ | ------------: | ------: | ---------------: | ------------: |
| Conservative |     158.054 s |     576 |          134,710 |             0 |
| Aggressive   |     149.900 s |     576 |          135,253 |             0 |

The observed loop-time difference is 5.16%. It cannot be attributed to KV migration:
neither arm executed any handoff. Token lengths and scheduling differ across runs, and
this is a single pair. Both runs exited with code zero. Core planner, coordinator, and
KV transport source hashes match; benchmark settings differ only in creation timestamp
and container prefix. The policy environment variable is recorded separately in each
workspace.

Evidence: [comparison.json](evidence/r001-aggressive-20261019/comparison.json).

The planned persistent-vLLM repeated benchmark is an experimental tooling improvement,
not a research contribution. Startup/reset time must remain separate from training-loop
performance. This experiment used independent identical initialization for both arms;
persistent reuse is not implemented here.
