English | [中文](R001-runtime-request-rebalancing.zh.md)

> Current accepted policy and results:
> [2026-10-19](../experiments/2026-10-19-r001-accepted-policy.md). Earlier sections
> below record historical stages.

# R001: Cost-aware rebalancing of already-running requests

Created: 2026-10-15; updated: 2026-10-16 (America/Vancouver). Status: experimental
implementation; live KV continuation and RL routing validated. Six-batch c16/c32 tests
did not establish additional migration speedup. No novelty claim.

## Mechanism diagram

![R001: Rebalance running work only when the critical path wins](R001.png)

Optimize batch completion time, not equal token counts.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R001.svg) | [PNG](R001.png)

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

Related-work review: pending. Implementation: uncommitted experimental code. Live
results: see the initial findings below; no live running migration observed. Update this
section with experiment IDs, negative findings and revised hypotheses.

## Proposed implementation locations and paper framing

Expected touchpoints: `areal/infra/queue_rebalance.py` (candidates/cost model),
`request_router.py` (leases/reservations), `areal/engine/vllm_ext/queue_rebalance.py`
(safe engine interruption), and `rollout_controller.py` (versions/recovery). These
locations now contain the experimental implementation described below.

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

## Implementation update (2026-10-15)

The opt-in policy now has `off`, `shadow`, `budget`, and `active` modes. The benchmark
flag is `--running-rebalance`; the internal environment variable is
`AREAL_RUNNING_REBALANCE`. Defaults preserve queued-only rebalancing.

Implemented safeguards include one candidate per decision, one attempt to move per
logical request, native engine-loop abort, source response acknowledgement,
version-pinned ownership handoff, retained version constraints after rejected admission,
bounded retries, phase-specific KV-budget checks, and prefix replay cost.
`running_rebalance.py` wraps the existing causal decode simulator; `request_router.py`
protects ownership and epoch transitions. The vLLM endpoint rechecks memory eligibility
before admitting the continuation.

The initial estimator optimizes the predicted critical tail of the live request set. It
checks whether training is already ready, but does **not** yet predict the future
training withdrawal deadline or quantify training slowdown uncertainty. These are
explicit limitations, not completed research contributions. The publication watchdog is
failure protection, not the removed fixed drain timer.

`benchmark/npu_share/analyze_running_rebalance.py` summarizes decisions and audits lease
overlap and handoff versions. It cannot establish reward equivalence or causal speedup.
Real-vLLM contract tests verify running cancellation and retained cumulative
tokens/logprobs; live validation results will be recorded separately.

For causal ablation, `--running-rebalance budget` applies the same inference-only KV cap
as `active` and records forecasts without running-request migration. The original `off`
and read-only `shadow` modes do not establish this new cap. Thus active versus off
measures a combined budget/scheduling change; use budget versus active to isolate
migration and repeat seeds before claiming a speedup.

## Initial live findings (2026-10-15)

Both three-batch runs completed without ownership audit violations, but neither
performed a running-request migration. Shadow rejected both decisions because no
concurrent training budget was reserved. The first active run also rejected both. An
initial hypothesis about offloaded actor storage was disproved by detailed logs: actor
weights and optimizer remained resident. The growth reservation exceeded free device
memory.

An intermediate (now superseded) reservation included actor restore bytes and parked
workspace bytes, in addition to optimizer, gradient, publication, training/inference
workspace and safety margin. It reserves future restoration without restoring the actor
or weakening the normal during-PPO residency requirement. Validation of this correction
is tracked separately; zero migrations is not a successful end-to-end migration
demonstration.

Shadow fixed/shared loops were 102.026/89.815 seconds. The first active loops were
100.917/114.981 seconds, with 68,839/67,390 generated tokens. These single runs cannot
establish R001 benefit, especially with no running migrations. Evidence:
[initial validation](../experiments/evidence/r001-initial-validation/).

### Follow-up budget diagnosis

Run `20261015-230757-0700_math_fsdp_c16_b3_77def7` completed all three batches: fixed
104.790 s, shared 88.361 s; 67,501/68,239 generated tokens. Both recovery decisions
still rejected migration, with zero ownership violations and zero outstanding leases.
The shared speed difference is not an R001 migration result because no running requests
moved.

The first recovery had 18,239,004,672 free device bytes but requested 27,649,693,696
reserved growth bytes, producing a zero KV budget. Dynamic KV was enabled and actor
restore bytes were zero. The restore-reservation change is defensive support for an
offloaded actor, not the cause or fix of this observed rejection. Whether the retained
actor workspace and configured growth allowance overlap requires allocator-level
evidence; do not credit cached bytes or reduce the safety reservation speculatively.
Running-request end-to-end migration and causal performance benefit remain unvalidated.

Evidence: `active-reserved-restore.json`, its summary and `budget-rejection.log` in the
initial-validation evidence directory.

## Phase-specific admission correction (2026-10-15)

The training-first reservation at shared inference recovery was an incorrect policy for
this experiment. Inference-only recovery now reserves inference workspace and safety
margin, not future actor restoration, optimizer, gradient or training workspace.
Existing training prefetch closes admission near batch completion, drains/aborts owned
requests, reclaims inference memory, and restores training state. Actual
training/inference overlap still uses concurrent_train().

R001 budget and active modes both use inference_budget() for a matched control. Each
device enforces its own limit and TP uses the minimum capacity. The server checks
phase-specific KV validity and full continuation headroom again at admission. It does
not unconditionally offload all actor memory: existing pressure-based inference
restoration retains resident storage when space permits. The earlier restore-reservation
implementation has been removed. Historical negative results above are retained as
evidence of the rejected policy.

### Inference-only budget validation (2026-10-16)

Run `20261015-235533-0700_math_fsdp_c16_b3_795d4f` completed both arms, three batches
each. Fixed/shared loops: 100.868/94.355 seconds; generated tokens: 68,403/66,134. Both
recovery budgets were admitted, reserving only 4 GiB for inference workspace and safety,
with KV budgets of 14,686,363,648 and 14,795,333,632 bytes (4,000 effective blocks). The
training-reservation rejection is resolved.

Both decisions returned no_predicted_gain: the source had 48 and six running requests
respectively, the destination had zero, and neither had queued requests. No running
request moved. Ownership violations and outstanding leases were zero. The 6.46% shared
loop reduction is not evidence of R001 migration benefit; generated token totals also
differed. Actual running handoff remains covered by contract/unit tests, not this live
run.

195 targeted regression tests passed, including per-device group failure and
offloaded-actor inference admission. Evidence: inference-budget.json,
inference-budget-summary.md, and inference-budget.log in the evidence directory.

## KV transport and batch-plan prototype (2026-10-16)

A two-NPU real-model probe transferred BF16 Qwen2.5-1.5B request KV over HCCL. For
computed contexts of 64, 256 and 1,024 tokens, payload sizes were 1.75, 7 and 28 MiB. KV
elements matched exactly and both source and receiver generated the same next 16 greedy
tokens, retaining the last sampled but uncomputed token. Warm packed transfer medians
were 3.927, 4.173 and 4.102 ms; prefix replay in the same Transformers eager probe took
63.267, 69.421 and 133.491 ms. These are not vLLM replay timings or full live-migration
costs. Each median uses four warm samples after one initial transfer; no concurrent
training traffic was tested.

New files: `areal/engine/kv_transfer.py` (transport primitive),
`benchmark/npu_share/kv_transfer_probe.py` (correctness/cost probe),
`areal/infra/batch_rebalance.py` (non-executing batch planner), and
`benchmark/npu_share/replay_batch_rebalance.py` (causal snapshot replay). The planner
scans one snapshot, compares replay and KV cost, evaluates several ordered candidate
subsets, limits destination capacity, and includes measured planning time. Unknown
ownership costs are an explicit sensitivity parameter. It never authorizes scheduler
execution (`execution_ready=False`).

On the saved 48-request snapshot, a 16-request replay plan predicted 0.155 s raw tail
reduction; about 0.087 s of planning left only 0.069 s net. A KV scenario with zero
extra ownership cost selected 16 requests with essentially the same tail prediction.
Adding 50 ms ownership cost selected ten candidates but only predicted 0.023 s gross
gain, less than planning cost, so it was rejected. The six-request snapshot predicted no
gain. These are model counterfactuals, not observed rollout speedups; the transport
probe's layout and 64-token block rounding are explicit assumptions when replaying old
snapshots.

The remaining vLLM work is request freeze/drain acknowledgement, fresh target block
allocation, sparse-page ownership import before scheduling, exact computed-token
restoration, and source release only after destination ACK. The current slab path
rejects stock kv_transfer_config, so a stock connector is not a drop-in solution. No
live KV handoff has been enabled.

[Raw probe and replay evidence](../experiments/evidence/r001-kv-prototype-20261016/summary.json).

### Reviewed cost accounting

The authoritative follow-up is `batch-replay-reviewed.json`; the earlier replay file is
preliminary. The revised model charges a serialized KV-transfer pause to both engines,
accounts for uncomputed prefill tokens, and preserves transfer ordering when evaluating
subsets. In the 48-request snapshot, replay-only selects 16 requests: predicted tail
falls from 11.576 to 11.421 seconds, but 0.103 seconds of measured planning leaves only
0.052 seconds of predicted net benefit. Adding KV alternatives does not improve the
selected tail; the zero/50-ms ownership cases spend about 0.173 seconds planning and are
rejected. The six-request snapshot still selects no migration. These are predictions,
not observed RL speedups. Planning time is host-load-sensitive. The current conservative
whole-engine pause protocol can erase the advantage of cheap KV transport; request-local
ownership transfer remains unimplemented. The final targeted regression suite passed 83
tests.

### Online KV handoff experiment

The experimental `--running-rebalance kv` mode adds request-local KV transport to the
shared arm. It retains the existing queued-request policy, then considers same-version
running requests only when the target has admitted inference memory and training is not
imminent. A causal batch-window forecast compares bounded subsets (at most 16 requests)
against leaving work in place. The initial whole-handoff bound is 100 ms per request;
subsequent observations can increase it. The predicted net benefit must exceed both 250
ms and twice the observed handoff cost. These conservative thresholds are provisional
and must be calibrated with online measurements.

The source stops scheduling only selected requests, retaining their slots and KV
ownership. Once their outstanding steps retire, a worker packs their blocks into an
immutable payload. A dedicated, prewarmed HCCL group transfers it to destination
staging. A one-use ticket binds the continuation identity, token-prefix digest, exact
computed count, geometry and route version. Only after target staging is acknowledged is
the source aborted through the existing token/logprob accumulation path. Native
destination admission allocates owned blocks, copies staged KV, and computes the
remaining pending token. Sparse pages remain pinned until worker ownership exists;
cancellation also retires pins.

The first implementation supports two TP=1 Qwen2 full-attention replicas with dynamic
KV, eager execution, prefix caching and speculation disabled, and unseeded generation
without penalties, minimum-token constraints or stop strings. Failure of an uncertain
distributed transaction stops the run rather than authorizing duplicate execution. The
mode is opt-in. No universal no-regression guarantee is claimed: forecasting and
communication themselves cost time, and predictions can be wrong. Acceptance requires
token/logprob correctness plus matched workload comparisons and a benefit case,
including regression cases where migration is rejected.

Online correctness probe passed: 34 generated tokens on the source, actual import of 91
computed KV tokens, and 256 total generated tokens identical to uninterrupted greedy
generation. Maximum logprob error was 0.0. The source packed 3.5 MiB in 11.57 ms; the
two endpoints measured HCCL transfer at 5.07/1.84 ms. These component times exclude the
full request handoff. Evidence:
`../experiments/evidence/r001-kv-live-20261016/correctness.json`. The runtime Ascend
scheduler replacement required attaching the freeze hook to the actual scheduler class;
the regression test covers this replacement.

### Conservative live scheduling gate (2026-10-16 follow-up)

The live adapter now evaluates only executable KV candidates with exact native
computed-token counts. The current transport stages the selected requests before a
common source-abort barrier. The forecast therefore charges the aggregate handoff
interval to both peers, including unmoved requests, rather than assuming that early
transfers immediately resume or that worker RPCs are free. This is a conservative
approximation, not a proof of non-regression.

Training readiness is checked before each selected export and again before the source
abort. If training becomes ready, staged destination tickets are discarded, source
requests are unfrozen, and routing reservations are canceled. Native running/waiting
membership is required; router-active requests alone are not a measurement of decode
concurrency. A new snapshot event records the inputs and preserves the actual rejection
reason for audit.

A flat decode-service regression test explicitly rejects moving requests merely to
equalize counts. Benefit must come from lower completion time after costs. No additional
speedup is established by these safeguards. In particular, the first six-batch c16
diagnostic completed without any live KV handoff, so its fixed/shared difference must
not be attributed to this feature.

### Reconcile admission and completion races without resampling

The c16 live snapshots exposed normal gaps between router leases and native engine
membership. A new lease may not have reached the engine yet; a finished attempt may
still await its router release callback. Rejecting the entire plan on either gap
prevented useful analysis.

The planner now treats a not-yet-admitted lease as waiting work, includes its expected
prefill/continuation in the forecast, reserves its target capacity, and never selects it
for KV migration. A completed attempt is excluded only when its history entry matches
the request ID and current full prompt length (original prompt plus retained generated
prefix). A shorter old attempt is not proof that the current continuation completed.
Live rows with ambiguous native membership still reject the snapshot. No additional RPC
or router-wide admission lock is introduced.

Replaying the five captured recovery snapshots after this correction completed all five
forecasts. All five still selected zero moves. This is counterfactual validation of
snapshot handling, not a measured scheduling speedup.

The six-batch live comparisons and their limitations are recorded in the
[2026-10-16 experiment report](../experiments/2026-10-16-r001-live-kv-scheduling.md).
