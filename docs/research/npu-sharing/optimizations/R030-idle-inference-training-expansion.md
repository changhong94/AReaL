# R030: Expand training onto idle inference devices

[中文](R030-idle-inference-training-expansion.zh.md)

## Status and classification

The latest (2026-10-25) budget version passed correctness validation. Shared took
475.848s versus fixed 462.487s (2.889% slower), with no actual mid-update switches and
16.728s of batch-entry round trips. Fewer switches have not yet beaten fixed; this is
one development pair, detailed below.

## Problem and hypothesis

Fast generation and slower training leave a dedicated inference NPU idle. Initialize
both DP=1 and DP=2 training engines before the RL loop. Normally train on one NPU; when
no runnable generation remains, use both. Benefit requires saved compute to exceed state
resharding, memory transitions and publication costs. CPU scoring is independent and may
continue.

## Implemented baseline and evidence

Single and dual actors use independent process groups. The inactive actor is parked in
DRAM; an atomic admission fence prevents new generation racing with selection.
Off-policy-blocked pending inputs are not runnable generation. Selection occurs at batch
boundaries, never during an optimizer update. DCP transfers model, Adam, scheduler and
version state between actor worlds, then returns to the original publisher. HCCL auto
port allocation avoids co-resident process collisions. Gradients are retired and
retained workspace mappings parked before offload.

Code: `areal/trainer/idle_training.py`, `rl_trainer.py`, `areal/engine/fsdp_engine.py`,
`areal/infra/controller/train_controller.py`, `areal/infra/workflow_executor.py`; opt-in
`AREAL_IDLE_TRAINING=1` affects only the shared benchmark arm. Current scope: local
two-NPU FSDP, inference TP1, Qwen-style verification, no auxiliary models/recovery.

Identical-generated-input training replay previously measured 365.893→259.446 seconds
over six batches (29.09% lower), excluding switching. The subsequent full code-RL test
used six batches, 32 questions/batch, eight answers/question, concurrency16, scoring16,
offpolicy1 and graph inference:

| Metric (seconds unless noted)        |   Fixed | Expanded shared |
| ------------------------------------ | ------: | --------------: |
| RL loop                              | 460.921 |         866.057 |
| Trainer initialization, outside loop | 180.789 |         299.103 |
| PPO compute total                    | 288.851 |         263.178 |
| Added switch-in/out total            |       0 |         433.920 |
| Generated tokens                     |  364652 |          349163 |

Only batches1 and6 selected DP2. Four handoffs passed sampled weight/moment, step,
scheduler and version checks. The loop regressed87.90%. The433.920 seconds includes
checkpoint export/load, state transformations, memory transitions, synchronization,
verification and cleanup; it is not an isolated disk-I/O measurement. Generated answers
differ in this full run; do not infer isolated compute benefit from its PPO totals.

[Detailed experiment and failures](../experiments/idle-training-20261023.md) ·
[compact evidence](../experiments/evidence/idle-training-20261023/analysis.json). No
formal repeated confirmation yet.

## Why checkpoints were used

DCP was a conservative correctness baseline for reshaping DP1↔DP2 weights and Adam
shards while preserving scheduler progress and the original publisher. It is not a
requirement of RL, FSDP role switching or the proposed research. Two separately
initialized training worlds need current optimizer state; copying inference weights
alone cannot continue equivalent training. The present return trip also reflects the
existing single-actor publisher architecture, not an unavoidable algorithmic
requirement.

## Relation to HybridFlow

[HybridFlow §5](https://arxiv.org/html/2409.19256v2#S5) describes training and
generation on the same device set, reusing actor weights with different parallel groups.
Its group layout minimizes redundant weight storage and confines all-gathers to micro-DP
groups. This differs from transferring complete Adam state between independent DP1/DP2
training engines. Its communication design does not require a disk checkpoint on each
transition. Hardware, model and operation differences prevent direct latency comparison;
the paper does not establish an expected latency for our prototype.

## Next design: direct state shards through RAM

1. During initialization, build a manifest of stable parameter names, state names,
   global shapes, dtypes, placements, shard offsets and optimizer-group metadata.
   Preallocate a bounded, reusable host-memory staging region after checking host
   capacity.
1. At a safe update boundary, synchronize the source. Write local weight and Adam shards
   into disjoint shared-memory ranges. A replicated tensor is written once. Also
   transfer persistent buffers, optimizer steps/hyperparameters, scheduler and version.
1. Retire source gradients, park workspace and offload source state. Restore the
   destination. Read only the destination shard slices and copy in place into its
   existing tensor storage. Preserve DTensor wrappers, parameter identity, optimizer
   objects, graph-sensitive addresses and memory-pool ownership.
1. Require all-rank validation and acknowledgements before reusing the staging region.
   Reject unsupported placement or dtype instead of guessing. Initial supported layout
   should be one-dimensional DP Shard(0)/Replicate; fail closed on partial transfers.

Explicitly managed shared-memory mmap avoids Python3.11 shared-memory resource-tracker
ownership problems. A `/dev/shm` mapping is RAM-backed, but host swapping must be
checked; merely moving `torch.save` to tmpfs retains serialization and reconstruction
costs and is not the intended endpoint. The inactive actor's DRAM backup and the staging
region coexist, so memory accounting must include both.

## HCCL option and publication redesign

Use HCCL for the remote-device portions when beneficial, while local overlapping-device
data can use host/shared-memory or an appropriate local-copy path. Do not naively put
the single actor and both dual actors into one three-rank HCCL group: two processes
share a physical NPU. Any transfer communicator and buffers must be initialized/warmed
before the loop and validated for that topology. Bucket transfers to bound staging
memory and avoid per-tensor RPC overhead.

Longer term, publish directly from the current training engine and keep DP2 active
across consecutive idle intervals. Avoid returning the entire optimizer state to DP1
merely to publish weights or end training. This requires checkpoint/evaluation/version
ownership changes; it is not part of the completed baseline.

## Validation and acceptance

First compare DCP versus direct RAM handoffs with identical saved inputs, including an
update on each side and all parameter/moment shard checks, steps, scheduler and
versions. Test odd shard sizes, replicated buffers, dtype mismatches and interrupted
transfers. Log source-copy, synchronization, reshard/import, memory switch, validation,
bytes and peak host/HBM memory separately. Then run the same six-batch complete
fixed/shared development pair, keeping logs, model, workload and scoring conditions
matched. Report initialization separately and include all role-switch overhead in the
loop. Keep the negative baseline. Do not promise a latency or global default until
measured positive end-to-end results exist.

## 2026-10-23: Asynchronous shared-DRAM implementation

A new prototype in `areal/engine/idle_state.py` now exports versioned local state shards
to preallocated `/dev/shm` mappings, without DCP, tensor serialization, or checkpoint
files in the handoff path. A background thread copies local shards only; no background
FSDP collective is used. Existing actor snapshot/mutation/unmapping fences join this
work before the source can change. Scheduler/group metadata is captured with the state.
Selection checks training progress; the target imports in place and verifies every
imported local byte, including moments and buffers. The two-batch NPU gate and
subsequent six-batch paired run are pending; do not claim a measured speedup yet.

Training-side slab support is reused through `AscendPoolBackend`/CaMem and
`TaggedMemory`, including weight/optimizer pools and parked workspaces. The new logical
snapshot is separate from a slab allocation backup: the latter has process-local
addresses and is cleared on restore. Direct reuse of slab CPU backups to remove
duplicate D2H is not yet implemented. Current background copies can be joined early by
publication/stats operations, so asynchronous initiation alone does not prove useful
overlap. Full-byte import verification is enabled in this experiment and its cost is
included.

CPU Gloo DP1→DP2→DP1 validation passed, including odd leading-axis shards, replicated
scalar buffers, optimizer moments/step and scheduler continuity. Another58
idle/staleness tests passed. The shared benchmark container gets64GiB tmpfs capacity
only when this feature is enabled; storage is reserved during initialization. The host
has no active swap. Fixed-arm configuration is unchanged.

An independent remote driver performs the correctness gate before the full comparison,
saves failures and restores externally managed resource occupancy. Driver state:
`/root/areal-development/idle-dram-20261023/chain-state.json`. Driver orchestration is
tooling, not a research contribution.

### Sandbox interruption and bounded upload recovery

The first NPU smoke run completed one DP1-to-DP2-to-DP1 round trip. All three rank-local
state imports passed bytewise verification. Its first-round switch cost was 27.02 s into
DP2 and 53.74 s back (80.76 s total), versus 204.94 s for the first round of the disk
baseline. These partial observations are not a completed six-batch performance result.

The run then aborted because the first test-file upload returned `sandbox not found`.
This occurred before test execution, not during training-state transfer. Existing
service logs cannot distinguish a transient missing route from permanent sandbox loss.
The benchmark adapter now retries only this specific idempotent upload error, at most
five delays totaling 5 seconds. It never retries an executed command or converts
infrastructure failure into a zero reward. Persistent failures still abort. Both arms
use the same recovery code. A 16-worker, 64-sandbox live preflight passed in 26.81
seconds. Retry events are recorded for auditing; this is benchmark reliability work, not
an R030 algorithmic gain.

The replacement correctness-then-six-batch chain is under
`/root/areal-development/idle-dram-20261023-retry`; The chain has completed; see the
results below.

### Completed NPU validation and six-batch reused-vLLM comparison

The replacement chain completed successfully. The two-batch smoke and six-batch pair
each exercised two DP1-to-DP2-to-DP1 round trips, with six successful bytewise
rank-local imports per run. Optimizer steps, scheduler state and sampled model
fingerprints matched at the handoffs. Both fixed and shared arms completed, and neither
triggered the new upload retry. The external driver restored `make-npu-busy`,
independently confirmed running. No benchmark repository logic manages that container.

| Six-batch metric                                                   |   Fixed | Shared DRAM |
| ------------------------------------------------------------------ | ------: | ----------: |
| RL loop, seconds (includes switching and full import verification) |  466.00 |      601.42 |
| PPO update total, seconds                                          |  292.92 |      262.65 |
| Training-engine switching, seconds                                 |       0 |      158.47 |
| Reported generated tokens                                          | 364,504 |     366,253 |

Shared was 135.42 seconds (29.06%) slower overall. Only B1 and B6 used dual-card
training. Switching into dual-card training cost 26.75 and 30.01 seconds; returning cost
51.24 and 50.47 seconds. Against the previous disk-based six-batch run, which also
switched at B1 and B6, switching fell from 433.92 to 158.47 seconds: 275.45 seconds /
63.48% less. This is a cross-run comparison with different generated answers, not a
controlled replay proving all gain is from the transport change.

The async snapshot join itself was only about 0.03 seconds per switch-in and 1.50/1.07
seconds per return. Most remaining switch cost is elsewhere in the measured
offload/onload/import/verification path; these numbers do not isolate each component.
Full-state verification remains enabled and counted. Restoring an old DRAM backup before
replacing it with the current state is still present. No claim of net RL speedup is
justified; R030 remains opt-in. vLLM reuse is development tooling, not a research
contribution or independent-engine formal confirmation.

Evidence: [analysis](../experiments/evidence/idle-dram-20261023/analysis.json),
[summary](../experiments/evidence/idle-dram-20261023/summary.json),
[driver state](../experiments/evidence/idle-dram-20261023/chain-state.json). Workspace:
`/root/areal-development/idle-dram-20261023-retry/comparison/20261023-221040-0700_code_fsdp_c16_b6_12e8c8/repetitions/01`.

### Switch-overhead audit

Code inspection found that the asynchronous snapshot is separate from the inactive
engine's CaMem CPU backup. The current path serially parks/offloads the source, restores
the destination's old backup, overwrites it with the latest snapshot, then reads every
imported tensor back to CPU for bytewise equality and SHA256. Snapshot export and slab
offload duplicate D2H; restoring the old backup and importing the latest snapshot
duplicate H2D. Returning to DP1 also concentrates the full verification on one process
and restores inference pools.

A CPU-only bounded-slab probe processed 18,524,573,512 bytes (the single-rank import
size): equality 0.80 s, `numpy().tobytes()` 8.51 s, SHA256 over a memoryview 26.78 s,
and the actual `sha256(tobytes())` pattern 33.21 s. These independent measurements are
not additive and are not an exact attribution of the prior NPU run. They establish
substantial CPU validation cost; the original switch is not a single DMA operation.

Aggregate per-rank import/verification/equality/hash timers and controller
source-offload/destination-restore/import timers were added for subsequent runs, without
extra per-tensor synchronization. Full validation remains enabled. Future phase
rendering includes separate scoring and training-engine-switch lanes when evidence
exists; previous images are unchanged. Scoring bars mean at least one active scoring job
(including sandbox lifecycle), excluding queue wait, not summed CPU occupancy.

### Direct parked-backup staging prototype

The destination now receives remapped CPU shards directly into its preserved CaMem
`cpu_backup_tensor` while offloaded, before onload. Device tensor pointers are used only
to locate byte intervals; no unmapped device memory is read. All targets and metadata
are validated before copying. CPU optimizer-step tensors are updated directly. Onload
restores current state once, without the previous stale restore plus overwrite. Source
export remains asynchronous; destination staging is still measured critical-path CPU
work, not a fully synchronized standby engine.

`AREAL_IDLE_FULL_VERIFY=1` is the default and validates complete state after restore.
Performance experiments may explicitly set it to `0` only after correctness passes;
version, optimizer step, scheduler and sampled state checks remain.
Full-verification-disabled events carry `full_verification=false` and
`verified_bytes=0`. The external chain at `/root/areal-development/idle-direct-20261023`
enforces full correctness first, then six-batch reused-vLLM performance comparison.
Results are recorded below.

### Direct-backup validation and performance result (2026-10-24)

Two-batch correctness passed two complete DP1→DP2→DP1 round trips, including all six
rank-local full bytewise imports. The subsequent six-batch reused-vLLM comparison
explicitly disabled full verification and retained metadata/sampled checks. Trace flags
confirm this distinction; `verified_bytes=0` in performance mode is not full-state
verification. Both arms completed with zero upload retries. The external driver restored
the busy container and confirmed running.

| Metric                              |   Fixed | Shared direct-backup |
| ----------------------------------- | ------: | -------------------: |
| RL loop seconds, switching included |  472.07 |               487.02 |
| PPO update seconds                  |  297.70 |               269.04 |
| Switch seconds                      |       0 |                33.83 |
| Reported generated tokens           | 372,574 |              360,932 |

Shared remains 14.94 seconds / 3.17% slower; no net improvement over fixed is
established. Two switch-ins cost 5.37/7.18 s and returns 10.46/10.82 s, at B1 and B6.
Against the preceding 158.47-second DRAM run with full verification, switch time fell
78.65%. This combines the direct-backup implementation, removal of full verification
from the timed path, and run variation; it must not be attributed exclusively to
eliminating a copy. Different generated tokens also preclude exact same-work compute
claims.

Measured destination HBM restoration was only 0.84–1.47 s per transition. CPU backup
staging RPCs were 1.63–1.77 s into DP2 and 2.60–2.69 s into DP1. In correctness mode,
complete verification RPCs cost 18.62/19.96 s into DP2 and 37.19/41.46 s into DP1. The
heavy CPU verification, not a single HBM copy, explains most previous switch delay.
Performance-mode verification RPC still performs bookkeeping/releases retained mmap and
took 0.61–2.09 s. Source export plus slab backup is still duplicated and destination
staging is not yet overlapped. R030 remains opt-in; full verification stays default on.

The new phase figures contain scoring and both training-switch lanes. Evidence:
[analysis](../experiments/evidence/idle-direct-20261023/analysis.json),
[summary](../experiments/evidence/idle-direct-20261023/summary.json),
[driver](../experiments/evidence/idle-direct-20261023/chain-state.json).

Workspace:
`/root/areal-development/idle-direct-20261023/comparison/20261023-235528-0700_code_fsdp_c16_b6_771a75/repetitions/01`.
Figures: `figures/2026-10-24_phases/overview.png` and `overview.zh.png`, plus request,
token and utilization families.

### Standby prestaging and overlap prototype

The controller now prepares the parked DP2 backup in a background future after the
primary snapshot is complete, overlapping publication/rollout where possible. It joins
the future before source snapshot overwrite, destination restore, and teardown. A
consumed standby is matched by optimizer/scheduler generation and relabeled to the
current publication version. Background failure is fatal. When no prepared standby
exists (including DP2-to-DP1), CPU backup staging runs concurrently with source
workspace parking/offload; destination HBM restore waits for both.

Source mmap objects are cached per path. Disabled full verification no longer rebuilds
the tensor inventory, and clearing pending metadata no longer tears down the source
mapping. This addresses suspected metadata/mapping overhead without claiming that
previous timings isolated mmap cost. Source export is still separate from slab offload;
HBM restore remains budget-dependent and sequential after source release.

Fifteen CPU/distributed tests passed, including a synchronization-based test that
requires staging and source offload to overlap. The external
correctness-then-performance chain is `/root/areal-development/idle-overlap-20261024`;
results recorded below. Development vLLM reuse and external resource cleanup remain
tooling, not algorithm contributions.

### Overlap result (2026-10-24)

Two-batch correctness passed all six rank-local full imports. The six-batch performance
pair disabled full verification while retaining metadata and sampled checks. Fixed took
**457.77 s**, shared **470.65 s** (shared **2.81% slower**), with 357,213/361,736
reported generated tokens. Both arms had zero sandbox upload retries. Total measured
switching fell from the previous 33.83 s to **20.41 s** (39.65% lower). This is a
cross-run observation, not an isolated causal estimate.

B1 switch-in/return were 3.09/6.66 s; B6 were 4.08/6.60 s. Both switch-ins consumed
prepared standby backups; their join waits were 0.010/0.008 s. Returns overlapped CPU
staging with source parking/offload, leaving 0.88/0.71 s exposed staging waits.
Disabled-verification RPCs took 0.008–0.010 s, compared with 0.61–2.09 s previously.
Caching mappings and avoiding unused inventory work were changed together, so individual
attribution is unavailable. Return snapshot waits remained 1.83/1.71 s.

The implementation improves the switch path but has not established net RL speedup.
Different generated outputs and CPU/memory contention remain confounders. Evidence:
[analysis](../experiments/evidence/idle-overlap-20261024/analysis.json),
[summary](../experiments/evidence/idle-overlap-20261024/summary.json). Workspace:
`/root/areal-development/idle-overlap-20261024/comparison/20261024-012354-0700_code_fsdp_c16_b6_8c4a50/repetitions/01`.
External cleanup confirmed the busy container running before the next authorized
candidate.

### Direct snapshot-copy candidate

Replace `target.copy_(local.cpu())` with `target.copy_(local, non_blocking=False)` into
the existing mmap-backed CPU tensor, followed by a device synchronization before
readiness publication. This removes an explicit temporary CPU allocation/copy in our
code; the Ascend runtime may still internally stage pageable memory. Immutable-source
and generation fences remain. Fifteen CPU tests passed. NPU full validation and
subsequent timed pair run under `/root/areal-development/idle-directcopy-20261024`;
results recorded below.

### Direct snapshot-copy result (2026-10-24)

All six full rank-local imports passed in the two-batch validation. The subsequent
six-batch pair took **459.50 s fixed / 468.17 s shared**, shared **1.89% slower**.
Switching totaled **20.69 s**, effectively similar to the previous 20.41 s in this
single pair. First return cost 9.00 s, final return 5.32 s. Snapshot copy timings showed
warm-up sensitivity: initial DP1 export 5.29 s, final DP1 export 1.56 s; first DP2
export 3.05 s, final DP2 export 0.93 s. Direct copying is therefore not a uniform
latency improvement. No net RL speedup has been established; output differences and run
variability remain relevant. Zero upload retries; cleanup confirmed busy running.

Evidence: [analysis](../experiments/evidence/idle-directcopy-20261024/analysis.json),
[summary](../experiments/evidence/idle-directcopy-20261024/summary.json).

### Read-only publication overlap candidate

A private NPU copy stream now exports the idle training snapshot. Existing producer
synchronization precedes export and stream-local synchronization precedes readiness
publication. Controller polling carries an explicit generation; it does not occupy the
worker RPC thread waiting for export. Only resident DP1, full-model, fixed-receiver XCCL
publication may skip the idle-export join at both publication sites. Optimizer prewarm
and previous inference-snapshot joins remain. Mutation, load/save, slab transitions, new
export and teardown keep hard joins.

The candidate is not a general concurrent training/publication API. Checkpoints may
legitimately join the snapshot; bandwidth competition and polling delayed behind
publication can reduce benefit. Eighteen CPU tests passed. Three-batch full validation
must observe an actual publication while copy is pending before six-batch performance is
allowed. Chain: `/root/areal-development/idle-publish-overlap-20261024`; results
recorded below.

### Publication-overlap result (2026-10-24)

Three-batch full validation passed six rank-local complete imports and observed
publication while export was still pending. The six-batch timed pair took **462.24 s
fixed / 462.56 s shared**, a **0.32 s / 0.07% shared slowdown**, effectively a
single-run tie rather than established speedup. Tokens were 356,520 / 358,695. Switching
totaled **21.17 s**; shared PPO totaled 263.03 s versus 291.93 s fixed. Shared
weight-publication scope fell to **4.35 s**, equal to fixed. Four intermediate
publications started with export pending.

Some cost moved rather than disappeared: shared previous-tail waiting totaled 13.48 s
(9.54 s in the prior direct-copy pair), and clear-batch scope rose to 5.00 s (2.05 s
previously). Different outputs and scheduling prevent attributing these changes solely
to the implementation. Switch improvement and publication overlap are verified, but
sustained end-to-end speedup remains unproven. Zero sandbox upload retries; busy cleanup
confirmed. Evidence:
[analysis](../experiments/evidence/idle-publish-overlap-20261024/analysis.json),
[summary](../experiments/evidence/idle-publish-overlap-20261024/summary.json). An
unchanged-code confirmation is running separately.

A further initialization-only candidate exports the preinitialized DP2 state once before
its initial offload. This warms first-use export without mutating training state; the
real PPO export overwrites it with a newer generation. Snapshot capacity is already
allocated. It may shift cold-copy/page-fault cost out of the RL loop, but is not free:
initialization cost must be reported. First-use latency could also reflect contention or
NUMA placement. Candidate chain: `/root/areal-development/idle-export-prewarm-20261024`.

### Unchanged publication-overlap confirmation

A second six-batch pair with identical implementation measured **457.64 s fixed / 450.49
s shared**, a **7.15 s / 1.56% reduction**. Switching totaled **17.82 s**; tokens were
362,900 / 351,983, about 3.01% fewer shared tokens. The two pairs therefore range from a
0.07% slowdown to a 1.56% speedup. This is evidence of approaching break-even and a
small observed benefit, not proof of a robust or isolated algorithmic gain. No failures
or sandbox upload retries occurred. Reuse remains development-only.
[Evidence](../experiments/evidence/idle-publish-confirm-20261024/analysis.json).

### Initialization prewarm result

Full correctness passed, then fixed/shared RL took 463.37/452.65 s (2.31% faster
shared), with 369,944/356,302 tokens. Switching totaled 15.31 s; initialization prewarm
cost 4.74 s. First/final return snapshot copies were 0.755/0.748 s and switch returns
4.43/4.55 s. Source output differs, so isolated causal gain is not established. Retain
the validated candidate as opt-in.
[Full comparison and limitations](../experiments/idle-overlap-20261024.md);
[evidence](../experiments/evidence/idle-export-prewarm-20261024/analysis.json).

## Follow-up: retain DRAM backing across HBM restores (2026-10-24)

User-proposed follow-up, recorded but not implemented or measured: H2D restore is a
copy, so retain the DRAM backing. On the next eviction, unmap/release clean HBM blocks
without another D2H copy; write back only dirty blocks. Apply to both inference and
training residency.

Correctness requires per-block validity/generation tracking: optimizer steps dirty
weights and optimizer moments; inference appends KV blocks and may modify partially
filled blocks; weight publication invalidates old weight backups. Immutable KV prefixes
and unchanged weights can reuse valid backing. Fence in-flight kernels, copies and
consumers before unmapping. Retained virtual addresses may be required by graph replay.
Preserve memory-budget accounting for the persistent DRAM copy, and compare D2H bytes,
exposed transition latency, peak DRAM and end-to-end time. Never discard newer HBM data
using a stale CPU backup.

Implement after the current safe mid-batch training handoff experiment; this is an
unvalidated idea, not an established result.

## In progress: safe mid-batch expansion (2026-10-24)

The preceding six-batch run selected DP1 for B2-B5 because inference was busy at the
initial selection. Fixed inference subsequently stopped at 132.99, 219.30, 283.28 and
348.81 seconds; training continued until 186.58, 250.98, 316.46 and 382.78 seconds. A
batch-entry-only decision misses these idle intervals.

The experimental `AREAL_IDLE_MID_BATCH=1` path divides one original PPO optimizer
minibatch into backward segments (`AREAL_IDLE_CHUNK_ANSWERS`, initially 32). Each
segment uses the same full-minibatch loss denominator. At a completed backward/RPC
boundary, atomically fence admission only if generation is idle. Save the accumulated
DP1 gradient prefix into shared DRAM, perform the existing state handoff, and compute
the remaining suffix with DP2. Add the sharded prefix after DP2 gradient reduction, then
clip and update exactly once. Do not increase PPO minibatch count to create switch
points. Chunking changes length-balanced packing and floating-point order, so
equivalence must be measured rather than claimed bitwise.

Correctness gates: distributed CPU tests compare accumulated gradients, Adam
parameters/moments/step across early/middle/late prefix lengths; NPU validation replays
the full original batch at unchanged weights before stepping, checks finite relative
gradient error, and verifies complete state transport. Performance runs disable full
replay only after passing the gate. Fixed benchmark arm retains its original path.
Mid-batch phase scopes include prefix export and state switch. No performance conclusion
yet.

Follow-up idea: low inference load, not only zero load, may justify expansion. Do not
act on utilization alone. Estimate remaining inference work, pause/migration cost,
available training work and expected acceleration. Active requests require an explicit
pause/drain/migration policy, version consistency and a completion guarantee. Keep this
separate from the idle-only experiment.

Implementation pointer for retained DRAM backing: `areal/engine/colocation_memory.py`,
residency transition currently clears `a.cpu_backup_tensor` after mapping/restoring the
allocation. A future clean-backing mode must change this ownership rule together with
dirty tracking and budget accounting, not merely remove the assignment.

Mid-batch implementation notes: controller items are question groups (eight answers per
item in this experiment), not individual answers. Segment planning converts the answer
target to an even group count. A failed first validation exercised no mid-batch handoff
and was rejected; it is not performance evidence. Split-engine PPO statistics are
exposed under `idle_prefix/` and `idle_suffix/` instead of overwriting prefix values
with suffix means; do not interpret either as a whole-batch average.

### Mid-batch validation rejection and diagnosis (2026-10-24)

The second run exercised B2 expansion after 32/256 answers, but the pre-update
full-batch gradient replay had relative L2 error 0.086578, above the 0.02 gate. The
update was rejected and the six-batch performance comparison was not launched. This is
not a correctness-validated implementation.

Review found different question placement and microbatch packing between per-segment
dispatch and full-batch replay. The aggregate error therefore conflates transfer,
segmentation and cross-DP numerical effects. Added diagnostics compare identical-layout
repetition, the DP1 prefix replayed with identical packing on DP2, the suffix with
identical packing, and the full-batch reference. They report norm ratio, cosine
similarity, best-fit scale and largest-error parameters. No attribution or threshold
relaxation yet. Workspace: `/root/areal-development/idle-midbatch-diagnostic-20261024`.

### Preserved packing fixes the frozen-input gradient gate (2026-10-24)

A sandbox-independent frozen-input experiment reproduced a 9.96% relative gradient error
when backward segments were repacked independently. Identical-layout suffix replay was
1.37%, and the identical packed prefix compared across DP1/DP2 was 1.16%. This isolates
packing/layout changes as the dominant discrepancy in this experiment, rather than a
missing prefix or a simple DP scaling factor.

The revised implementation creates the ordinary PPO microbatch plan once on DP1, retains
every packed item and its padding, and dispatches intact items across DP2 after the
boundary. Odd final collective rounds use a zero-loss dummy; real answers are covered
exactly once. All segments retain the original full-batch denominator and perform only
one optimizer step. The plan resides in shared DRAM and has token/version checks.
Implementation: `areal/engine/idle_plan.py`, `fsdp_engine.py`, and
`areal/trainer/idle_training.py`.

The repeated frozen-input run passed the unchanged 2% gate with **1.3121%** relative
gradient error. Identical-layout suffix replay was 1.2581%; identical-prefix cross-DP
replay was 1.0205%. Both ranks completed the suffix and optimizer update. Evidence:
`/root/areal-development/idle-frozen-plan-20261024/events`. These are initial-weight
offline comparisons of the same saved input, not an integrated RL or performance result.
Actual residency/state handoff validation is still required before the six-batch timing
comparison.

Two independent interruptions were also diagnosed: a pure KV unmap incorrectly applied
an allocation reserve test (now the reserve check only applies when mapping new pages),
and a sandbox launch returned an unexpected EOF before readiness. Neither failed run is
counted as performance evidence.

The first integrated preserved-plan run passed the actual B2 gradient handoff at 68/256
answers (188 remaining), with 1.5963% relative error on both ranks. B3 then failed a
separate resident memory-budget check: after reclaiming inference storage, the policy
reverted to gross requirements despite confirmed reusable actor cache. The fix re-reads
allocator evidence after reclaim and checks incremental requirements only for fully
resident actors. Stale credits are never reused; parked/restorable allocations retain
the original check. Regression tests cover valid and invalidated credits. The six-batch
timing run remains gated on a complete integrated validation. Workspace:
`/root/areal-development/idle-midbatch-plan-v2-20261024`.

The preserved plan supersedes the earlier even-question-group segmentation description:
boundaries now fall between intact original microbatches, so answer counts per segment
may differ from the target. Selected plans do not copy the unused full source batch to
HBM. This does not alter the loss or answer coverage.

Integrated three-batch validation subsequently completed: B2 expanded after 32/256
answers, the unchanged full-gradient gate measured 1.2033% on both ranks, and nine
rank-local full state import checks passed. B3 passed the post-reclaim budget recheck
and completed. No sandbox upload retries occurred.
[Correctness evidence](../experiments/evidence/idle-midbatch-plan-20261024/correctness.json).
Six-batch warm-session fixed/shared comparison was then launched with full diagnostics
disabled; timing results are pending. Validation timing includes expensive replay and
byte checks and is not performance evidence.

Six-batch result: 458.79 s fixed / 496.32 s shared, an 8.18% slowdown despite successful
B2-B5 mid-batch expansion. Correctness passed; net performance benefit did not. Keep
opt-in. [Results and limitations](../experiments/idle-midbatch-plan-20261024.md).

## Retained DRAM backing prototype (2026-10-24)

`AREAL_RETAIN_DRAM=1` enables separate retained backing records for FSDP
weights/optimizer allocations and vLLM weights. Workspace and KV remain conservative
because their mutation lifecycle is not covered by these hooks. Mapped allocations keep
retained copies outside the allocator's `cpu_backup_tensor` field, preserving existing
cache-parking invariants. Records are keyed by allocation identity, handle and mutation
generation, not merely policy version. Writers join pending copies and invalidate before
optimizer/load/publication/reset writes. CPU staging updates the parked backing which
becomes authoritative on restore.

The existing quiescent training snapshot boundary starts a background full-allocation
D2H backup; inference starts its weight-only backup when execution resumes after
publication. Read-only publication/inference may overlap the copy. Unmapping and writers
join outstanding work; a failed copy poisons residency and cannot authorize unmapping.
Clean blocks reuse backing; dirty/new blocks have a synchronous fallback if not
prepared. Discarded blocks invalidate backing. Retained mapped bytes are included in
CPU-backup accounting. Full validation compares every reused raw allocation byte before
unmapping; performance mode disables that diagnostic copy. Logs separate copied, reused,
verified bytes, background copy time and exposed joins.

This duplicates some D2H traffic with logical-state snapshots and therefore is not
assumed beneficial. There is no KV-clean-block optimization yet, no new speedup claim,
and no default enablement. Forty-one CPU/state tests passed. Integrated full validation
and gated six-batch warm-vLLM fixed/shared comparison:
`/root/areal-development/idle-retained-20261024`.

### Retained backing validation, 2026-10-24

The first retained-backing validation passed raw backing equality and the mid-batch
gradient replay (relative L2 1.225%), but stopped at B3: training prefetch called the
gross-budget path even though the actor was already restored. The available
22,565,756,928 bytes were compared with a gross 25,502,210,048-byte requirement,
ignoring 3,741,319,168 bytes of verified reusable backward workspace. Prefetch now
requests guarded resident accounting; the policy still checks zero restore/parked bytes
and re-reads allocator evidence after reclaim. No safety margin or validation threshold
changed. The targeted prefetch/pressure suites passed 36 tests.

The corrected three-batch run completed in
`/root/areal-development/idle-retained-v2-20261024/correctness/20261024-151511-0700_code_fsdp_c16_b3_d082c6/repetitions/01`.
Nine rank-local full state imports passed. B2 switched after 34 of 256 answers (16
intact microbatches); both ranks passed full gradient replay at relative L2 1.15893%
(unchanged 2% gate). Raw backing verification covered 165,842,780,160 cumulative reused
bytes, including repeated transitions. Twelve background copy events copied
102,164,856,832 bytes in 6.7205 summed rank-local seconds; joins exposed only 0.000138
summed seconds. These are diagnostic-run cumulative measurements, not steady-state
throughput claims. The six-batch performance comparison disables full verification while
retaining production guards.

### Retained backing performance, 2026-10-24

- Workload: code RL, FSDP, six batches, 32 questions/batch, eight answers/question,
  question concurrency 16, scoring concurrency 16, offpolicy 1, graph inference, NPUs
  1/2.
- Development reused-vLLM run; one fixed/shared pair, not a formal repeated significance
  test.
- Fixed loop: 456.480733 s. Shared loop: 484.171510 s. Shared slower by 27.690778 s
  (6.066%).
- Shared summed switch-in/return: 62.518451 s; prior mid-batch candidate 68.209008 s.
  Prior shared loop 496.320186 s, but run-to-run generation differs, so this is
  contextual evidence rather than an isolated causal estimate.
- PPO scopes: fixed 290.088308 s; shared 264.992539 s including 26.729148 s nested
  mid-batch switches. Shared PPO excluding nested switches: 238.263391 s. Do not add
  nested scopes twice.
- Reported generated tokens: fixed 361000; shared 364037 (+0.84%).
- Full verification OFF in performance run: 18 metadata-only rank-local imports, no
  diagnostic gradient replay. Sandbox upload retries: zero in both arms.
- Initialization (excluded from loop): fixed 180.705436 s, shared 307.683273 s.
- Correctness preceding the run: nine full state imports; B2 full gradient replay
  relative L2 1.15893%; raw retained backing equality passed. Targeted tests: 41
  backing/state tests and 36 prefetch/pressure tests.
- Outcome: mechanism works and reduces copying, but overall performance remains worse
  than fixed in this sample. Keep experimental/opt-in. A subsequent candidate coalesces
  the DP2 suffix, preserving original microbatches and one optimizer step; its outcome
  must be reported separately.

Workspace:
`/root/areal-development/idle-retained-v2-20261024/comparison/20261024-153837-0700_code_fsdp_c16_b6_66760f/repetitions/01`.
Figures: workspace `figures/`.

### R030 retained backing and suffix experiment, 2026-10-24

The requested retained DRAM backing is implemented and fully validated for tracked model
weights and optimizer state. Clean backing copies survive H2D and are reused for
unmapping; dirty generations are asynchronously refreshed and fenced before any
mutation/unmap. KV cache and arbitrary workspaces remain conservatively managed because
their mutation lifecycle is not tracked. The feature remains experimental and opt-in
(`AREAL_RETAIN_DRAM=1`, with the existing idle-training flags). No global defaults were
promoted.

## Results

| Candidate                                  | Fixed loop (s) | Shared loop (s) | Shared slowdown | Shared switch total (s) |
| ------------------------------------------ | -------------: | --------------: | --------------: | ----------------------: |
| Previous mid-batch implementation          |         458.79 |          496.32 |           8.18% |                   68.21 |
| Retained backing + prefetch accounting fix |         456.48 |          484.17 |           6.07% |                   62.52 |
| Retained backing + coalesced DP2 suffix    |         463.37 |          487.96 |           5.31% |                   68.06 |

Each row is a separate development reused-vLLM pair, with a new fixed reference. These
are not matched-answer replays or formal repeated trials. The latest token counts are
fixed 362190, shared 371859; timings include switches, exclude initialization. Neither
new candidate beats fixed. Suffix coalescing has no established end-to-end gain: its
shared loop is 3.79 s slower than the retained-only run, while fixed also varies by 6.89
s. Keep the coalescing code within the opt-in prototype, not as a general performance
claim.

Latest shared PPO excluding nested switching: 232.74 s versus fixed 293.10 s. However
68.06 s of switching plus other shared-path overhead offsets the compute gain.
Generation completion occurs after 66-70 answers have been processed on the single actor
in B2-B5, so expansion uses only the remaining 186-190 answers; actual expansion
opportunities differ between runs.

## Validation and evidence

- Retained candidate: three full batches; nine full rank-local state imports; gradient
  replay relative L2 1.15893%, unchanged 2% threshold.
- Coalesced suffix: three full batches; nine full imports; gradient replay relative L2
  1.15500%. Odd/even suffix CPU tests preserve plan ranges, one optimizer update and
  prefix ownership.
- Targeted suites: 41 backing/state tests, 36 prefetch/pressure tests, 24
  controller/plan/gradient tests passed. Static checks passed on targeted new files.
- Performance: 18 metadata-only imports, no full gradient replay, no sandbox upload
  retries in either arm.
- Latest entire shared process traces (including initialization): 309,418,000,384
  cumulative bytes reused, 188,609,462,272 bytes refreshed in background (11.897 summed
  rank-local seconds), observed backup joins 0.000217 summed seconds. These cumulative
  values do not imply that all copy work is free: producer fences, CPU staging and
  bandwidth contention still contribute elsewhere.
- Initial failed retained attempt: B3 prefetch ignored verified resident workspace, now
  fixed with state-checked incremental budgeting; no safety reserve was reduced.

Latest comparison and four figure families:
`/root/areal-development/idle-retained-suffix-20261024/comparison/20261024-164250-0700_code_fsdp_c16_b6_2b61e5/repetitions/01`
(`figures/`). Retained-only comparison:
`/root/areal-development/idle-retained-v2-20261024/comparison/20261024-153837-0700_code_fsdp_c16_b6_66760f/repetitions/01`.

The external driver completed cleanup and verified `make-npu-busy` running.
Busy-container handling is not part of repository benchmark code. No commit or push
performed.

A next research direction is a measured net-benefit gate for mid-batch expansion, using
remaining original microbatch work and observed round-trip/gradient costs. It is not
implemented or claimed in this experiment.

## 2026-10-25: early shard preparation and independent rank restoration

Both directions now prepare immutable snapshot shards in background CPU threads. Each
destination waits for its own preparation and the release token for its physical device,
then restores and verifies independently. All source releases and destination
completions are joined before distributed inspection or training. After DP2 returns to
DP1, the parked DP2 actor is prepared from its latest snapshot during publication and
the next primary backward pass.

Implementation: `areal/engine/idle_handoff.py`, `idle_state.py`, `fsdp_engine.py`,
`areal/trainer/idle_training.py`, and `areal/infra/controller/train_controller.py`.
Snapshot generations, unique transfer tokens, visibility-namespace device identities,
atomic release files, failure signaling, and bounded waits protect the pipeline.
Snapshot mappings remain pinned until readers finish. Scope remains local
adaptive-memory DP1/DP2; the ordinary onload path with implicit barriers is unsupported.

Validation: 34 primary and 14 supplementary CPU tests passed (overlapping suites). Three
NPU batches exercised nine full rank-local state imports, sixteen retained-backup
verification events, and two gradient replay checks. Relative gradient L2 error was
1.132%, within the existing 2% gate, not bitwise identity. Performance used
metadata-only checks for eighteen imports.

The six-batch reused-vLLM code RL comparison retained FSDP, 32 questions/batch, eight
answers/question, concurrency16, reward concurrency16, offpolicy1 and inference graph
mode. Fixed took 461.578s; shared took 492.147s, slower by 30.569s (6.623%). Reported
tokens: 364419/363288. Total handoff time fell from 68.062s to 65.035s (3.027s, 4.45%),
without end-to-end improvement; previous shared time was 487.958s. Different generated
answers, packing and run variation prevent attributing all differences to this
optimization.

B2 expansion improved 7.398->6.641s: target preparation finished about42s before
handoff; physical card2 began restoration at4.127s and was ready at5.368s, card1
at6.613s, final completion6.641s. B2 return regressed5.844->6.051s: latest snapshot
readiness and full single-rank shard preparation remain critical.

The first performance attempt failed in fixed scoring after six exhausted
sandbox-not-found upload retries and is excluded. A16-concurrency32-sandbox health probe
passed before the successful complete rerun; that pair had zero upload retries. Earlier
per-card diagrams confused target rank with physical device: rank0 maps to card2, rank1
to card1; new plots correct this.

Evidence: [timing](../experiments/r030-pipeline-20261025/timing-analysis.json),
[correctness](../experiments/r030-pipeline-20261025/correctness.json),
[comparison](../experiments/r030-pipeline-20261025/comparison.json). Temporary switch
plots are Chinese-only; no plotting script was added.

### Mid-update switch time budget (2026-10-25)

Before fencing inference at a completed backward segment, predict net savings as
remaining DP1 time minus remaining DP2 time minus both complete handoffs. Accept only
above the configured margin; the existing inference-idle fence must still succeed.
Rejection does not mutate scheduling or training state. Batch-entry selection is
unchanged.

Work uses input_ids element counts from the preserved microbatch plan. DP2 sums the
heavier microbatch in every synchronized round, including odd-plan dummy rounds. This
token proxy cannot capture every sequence-length, communication or cache effect. Rates
use measured segment wall time over work in the current and previous three batches. DP2
suffix measurements include gradient import and optimizer overhead. Costs use the
maximum observed complete latency per direction, including gradient export on entry and
inference restoration on return.

Cold-start priors are a 1.5 speedup and a 15-second round trip, not validated
guarantees. No exploratory switch is forced when all candidates are rejected.
AREAL_IDLE_SWITCH_BUDGET defaults to 1 within the opt-in mid-update feature; 0 enables
ablation. AREAL_IDLE_SWITCH_COST_SECONDS=15, AREAL_IDLE_SWITCH_SPEEDUP=1.5 and
AREAL_IDLE_SWITCH_MARGIN_SECONDS=0 control priors and the acceptance margin.
idle_training_switch_budget records estimates, sample count, measured/prior sources and
rejection reasons. Implementation: areal/trainer/idle_budget.py, idle_training.py and
areal/engine/idle_plan.py. No new NPU performance comparison has yet validated this
change.

#### Budget validation result (2026-10-25)

All 25 CPU tests passed, including rejection before touching the inference fence.
Three-batch NPU correctness validation passed nine full shard imports and six state
transfers. Both ranks' gradient replay relative L2 was 1.3485%, below the existing 2%
gate. Diagnostic priors of 0.001 seconds and 2x speedup exercised one accepted
mid-update handoff; these are not performance defaults. NPU rejection coverage remains
pending.

The six-batch reused-vLLM comparison failed after three fixed batches: task 99 / session
817 encountered sandbox not found on six scoring-script upload attempts, triggering
exact-workload rejection. Shared never started. Partial fixed output contains 1057
completed answers and 252341 output tokens, excluding incomplete/cancelled answers. No
complete performance comparison or reduction in switches is established. No additional
parameter search was launched. Evidence: `../experiments/r030-budget-20261025/`.

### Default time budget: complete comparison after sandbox repair (2026-10-25)

Following the three-batch full correctness validation, sandbox Runtime.Launch received
an independent bounded application-start wait (infrastructure repair, not an R030
contribution). All 32 lifecycle probes at concurrency16 passed. The reused-vLLM
six-batch pair then completed without upload retries, with full verification disabled
for timing.

Fixed:462.487s; shared:475.848s, 13.362s/2.889% slower. Reported tokens:368059/351704.
Shared PPO284.300s versus fixed294.825s saves10.525s, insufficient for16.728s handoffs
plus other costs. B1/B6 selected DP2 at batch entry; B2–B5 remained DP1. Of28 budget
decisions,27 rejected and1 permitted but failed the inference-idle gate. No mid-update
handoff occurred. Every rejection was followed by a DP1 segment without an intervening
handoff. All decision overhead is in loop time; arithmetic was not separately
instrumented.

Prior shared492.147s/handoffs65.035s versus current475.848s/16.728s is an observational
improvement of16.298s/48.307s. Shared PPO increased from236.465s to284.300s,
illustrating the compute acceleration forgone. Different outputs and the sandbox repair
prevent attributing cross-run differences solely to the policy. Without DP2 suffix
samples, speedup/round-trip estimates retained1.5x/15s priors, not fully calibrated
optimal decisions. Diagnostic correctness covered a real accepted switch; this
performance run covered rejection.

One development pair, not formal confirmation. Existing answer/question reward
scheduling differences remain. Evidence, per-decision predictions, reasons and observed
subsequent PPO times: `../experiments/r030-budget-fixed-sandbox-20261025/`. Observed
residual time is not counterfactual savings. No tuning search or extra long experiment
was added.

## Agreed implementation sequence (2026-10-25; pending validation)

1. **Pre-training selection and preparation (current work).** Near batch readiness,
   choose DP1 or DP2 directly using runnable inference, historical training timing, and
   preparation state. Do not restore DP1 as a mandatory intermediate. Prepare current
   weights, optimizer, and scheduler state in the idle engine's DRAM backing. Compare
   incremental critical-path costs; do not charge common preparation twice or sum
   overlapping preparation. Validate correctness before a six-batch reused-vLLM
   fixed/shared comparison.
1. **Recalibrate mid-batch expansion after stage 1.** Retain safe microbatch boundaries
   and measure the remaining preparation, gradient transfer, and eventual recovery costs
   of the new path. Earlier preparation does not prove all round-trip costs disappear.
   Replace obsolete full-transfer priors with observed exposed costs.
1. **Helper computation when expansion is rejected, after stage 2.** Evaluate assigning
   outstanding complete microbatches to the idle card while the primary owns the
   optimizer and performs the only parameter update. A helper must use the same training
   weights and return gradients. Validate unique sample ownership, global normalization,
   gradient aggregation, communication cost, and deadlines for training and subsequent
   inference. This is a future candidate, not an implemented feature.

Each stage requires separate correctness, cost, and end-to-end evidence. Predicted
benefit is not a measured result.

### Direct pre-training selection prototype (2026-10-25; validating)

On the previous-tail executor, wait until the batch is nearly ready and prepare only the
selected DP1 or DP2 engine. DP2 does not first restore DP1. Runnable inference selects
DP1 without forced waiting or migration; later safe-boundary budget decisions remain
available. The rolling entry model uses exposed preparation and compute timing from the
current and previous three batches, charging incremental preparation and publication
costs. Missing observations are labeled priors. Batch timing is not yet normalized by
packed microbatch shapes.

The latest trainer remains authoritative. Its asynchronous DRAM export updates the
inactive engine's parked backing. Next entry may choose either engine; choosing the
source reuses its own backing. DP2 no longer returns to DP1 HBM for saving or
publication. It materializes a model-only snapshot from complete DRAM, releases training
HBM, and loads both vLLM replicas. DP1 retains its existing HCCL publication. This
publication change is included in timing and must not be attributed solely to admission
decisions. Scope remains local two-NPU FSDP full-parameter text models; parameter-only
inference publication does not provide general mutable-buffer support.

A DP2→DP1→DP2 diagnostic sequence verifies weights, optimizer, and scheduler state.
Forced selection requires full verification and is disabled in performance runs.
Evidence root: `/tmp/huawei/areal-development/idle-entry-20261025`. No performance
conclusion yet.

### Direct-entry correctness gate (2026-10-26)

The three-batch diagnostic completed with entry choices DP2, DP1, DP2. Seven full
shard-state imports passed, including weights, optimizer state and scheduler progress.
Batch 2 also exercised a mid-batch DP1-to-DP2 expansion: both ranks' gradient replay
relative L2 was 1.2525%, below the existing 2% gate. Both nonterminal publications were
driven by the authoritative DP2 DRAM state without restoring primary HBM. Seven
asynchronous standby-backup stage events were recorded. This validates the exercised
paths, not every topology or arbitrary model.

An early-preparation bug was repaired: pausing new admission must still harvest
already-running scoring completions. The regression verifies completion delivery while
queued work remains paused. Initial failed attempts are retained. Six-batch timing runs
disable full checks and forced entry choices. Performance results are pending. Evidence:
`/tmp/huawei/areal-development/idle-entry-20261025/retry-2/correctness/20261025-233928-0700_code_fsdp_c16_b3_6cb692/repetitions/01`.

### Six-batch direct-entry comparison (2026-10-26)

One reused-vLLM development pair completed: fixed 463.192s; shared 466.736s, shared
slower by 3.543s (0.765%). Reported output tokens were 360875 and 359324. The workload
used 192 questions, eight answers each, concurrency16, offpolicy1, graph inference, FSDP
and scoring concurrency16. Full state verification and forced entry selections were
disabled during timing.

Shared entry worlds were \[2,1,1,1,1,2\]; no mid-batch expansion occurred. Preparation
service times were \[1.947,1.810,0.280,0.202,0.121,4.220\] seconds; these can overlap
scoring and must not be summed as exposed loop overhead. First-batch DP2 publication
took 8.180 seconds without restoring primary HBM. The terminal batch skipped unnecessary
publication. Eight lightweight import checks completed.

Against the previous shared 475.848s run this is 9.113s shorter, but different generated
answers and multiple path changes preclude causal attribution. This run does not
establish an advantage over fixed allocation. The next agreed tasks remain incremental
mid-batch budgeting, then optional helper computation when expansion is rejected.
Evidence: `../experiments/r030-direct-entry-20261026/`; full charts and logs:
`/tmp/huawei/areal-development/idle-entry-20261025/retry-2/comparison/20261026-000110-0700_code_fsdp_c16_b6_6d5887/repetitions/01`.

### Incremental mid-batch budget and independent helper prototype (2026-10-26; validation pending)

The old switch budget required both entry and return observations, even after mandatory
DP2-to-DP1 return was removed. The latest six-batch run rejected all 28 probes with a
constant 15s charge; peak estimated compute savings in B2-B5 were 14.496, 13.693, 13.344
and 13.649s. The revised estimate charges measured entry (6s explicit cold prior) plus
positive DP2-minus-DP1 publication exposure; terminal publication is skipped. The 6s
prior is a provisional estimate, not a proven bound.

If expansion is rejected, an opt-in independent DP1 worker on the second device may
process an exclusive interval of the preserved packed plan. Primary retains earlier
gradients, executes the remaining interval, imports helper gradients exactly once, and
performs the only clipping/optimizer/scheduler update. Both use the original full-batch
denominator. Model and buffers refresh from the immutable authoritative DRAM snapshot;
helper optimizer state is not synchronized or stepped. The initial implementation still
allocates an ordinary FSDP actor and parks its optimizer; it is not yet a memory-minimal
helper. Never run only one rank of a DP2 FSDP group as an independent helper.

Helper admission compares predicted remaining single time to max(primary work, model
restore + helper work) plus gradient export/import costs. Restore and merge priors are
each 3s until measured. A forced diagnostic requires full verification and cannot bypass
inference-idle fencing. After helper release, fixed inference storage is restored before
DP1 HCCL publication. Whole-batch compute samples with helper work are excluded from the
single-engine entry-rate model. No performance claim yet; evidence root
`/tmp/huawei/areal-development/idle-incremental-helper-20261026`.

### Independent helper: correctness and development retry (2026-10-26)

The three-batch full check exercised DP2 -> DP1 with independent helper gradients ->
DP2, with five full state-import checks. Merged gradients had relative L2 error 1.19757%
against replay of the original complete microbatch plan, below the existing 2% BF16
tolerance. This is tolerance-based validation, not bitwise identity. Helper and primary
own disjoint intervals and use the full-batch denominator; only the primary performs the
optimizer step.

The first six-batch fixed arm completed in 453.126 seconds. The shared arm failed after
its B2 gradient merge because a scalar RPC timing result was incorrectly treated as an
iterable. It is not a performance result. The scalar handling and regression test are
fixed. The shared retry has identical input hashes and experimental settings to the
retained fixed arm; comparison provenance must disclose that reuse. Evidence root:
`/tmp/huawei/areal-development/idle-incremental-helper-20261026`.

The next candidate, `AREAL_IDLE_HELPER_LEAN=1`, skips Adam-state preallocation for the
backward-only helper. The helper never steps its optimizer, so these states need not be
moved. It remains disabled by default pending full NPU correctness verification and
end-to-end timing. No improvement is claimed yet.

The six-batch shared retry completed: fixed 453.126 s versus shared 451.901 s, a 1.226 s
(0.271%) saving. One development comparison does not establish a repeatable gain.
Reported token counts were 357751 and 360952. Shared B2-B4 delegated independent
gradients, B5 expanded mid-update in 6.382 s, and B1/B6 entered DP2 directly. By B5 the
B1 DP2 publication measurement had expired from the three-batch window, incorrectly
making the missing cost zero. The next revision retains the latest known cost per path,
explicitly labeled last_known_fallback, with recent observations taking priority. New
performance validation is pending. Compact evidence:
`../experiments/r030-incremental-helper-20261026/`.

The lean helper passed a three-batch full validation (forced DP2/DP1+helper/DP2): five
complete state imports, gradient replay relative L2 1.22074% within the existing 2%
tolerance. Memory records after helper backward/export show zero actor-optimizer
allocation. The retained-cost fallback now also applies to pre-training selection;
compute observations still expire normally, and terminal selection does not charge a
skipped publication. Two development comparison pairs are running, with the second
additionally enabling lean helper state; no lean performance claim yet.

### Retained-cost and lean-helper development pairs (2026-10-26)

Both complete six-batch comparisons now finish faster in shared mode. Pair 1 (retained
publication-cost fallback) is 460.517 s fixed versus 443.352 s shared, saving 17.166 s
(3.727%). Pair 2 additionally enables lean helper state: 469.806 s versus 452.697 s,
saving 17.109 s (3.642%). Generated tokens are 370584/363299 and 379179/369365
respectively, so these are workload-dependent development observations, not an isolated
causal estimate or a formal repeated-trial claim. Each arm resets and verifies initial
inference weights; services are reused.

Pair 1 rejects the late B5 expansion after retaining the measured 7.496 s extra
publication cost, and uses the independent helper instead. Pair 2 takes an early B2
expansion (7.305 s) and uses helpers in B3-B5. Lean helper preparation is 1.901-2.311 s
and release 0.973-1.049 s, versus 2.805-3.059 s and 1.398-1.437 s in pair 1. These
different execution paths prevent attributing the whole loop difference to lean state.
Gradient export still costs about 3.5 s.

A final optional candidate (`AREAL_IDLE_COMPARE_HELPER=1`, default off) compares
predicted net gains of expansion and independent helper execution even when expansion is
profitable. It preserves the inference-idle fence and unique optimizer step; prediction
events are not evidence of completed helper work. Full correctness for the underlying
helper and state-transfer paths has passed; the combined chooser has 43 CPU checks
passing across budget, entry, helper, and controller tests. Its end-to-end performance
is pending. Evidence:
`../experiments/r030-incremental-helper-20261026/budget-lean-comparisons.json`.

### Joint execution choice: completed development result (2026-10-26)

With retained costs, lean helper state, and joint expansion/helper choice enabled, fixed
took 463.256 s and shared 435.637 s: 27.619 s (5.962%) less within the six-batch
training loop, including in-loop preparation, handoff, publication, and final save.
Shared generated 365276 tokens versus fixed 356852 (2.36% more). B1/B6 entered DP2
directly; B2-B5 completed independent helper work; no mid-batch expansion occurred. The
helper preparation/export/import/release measurements are preserved in
`joint-choice-comparison.json`. Prediction events and confirmed completions are recorded
separately.

This is a reused-vLLM development comparison, with verified initial weight resets.
Shared includes existing inference and reward scheduling improvements, so the full gain
is not attributable solely to R030 or the new chooser. Initializing training engines
took 179.632 s fixed and 391.328 s shared. Including initialization and the loop gives
642.888 s versus 826.965 s: this short run is not faster from a cold start. Those sums
still exclude service startup/reset/reporting. Longer-run amortization and an
independent-start formal comparison remain unproven.

Three successive candidate comparisons show loop savings of 3.727%, 3.642%, and 5.962%;
the policies and generated workloads differ, so these are not three identical
statistical repetitions. Helper and joint-choice flags remain opt-in. The remaining
measured costs include roughly 3.6-4.4 s helper gradient export and expensive DP2
publication. Do not extrapolate these six-batch results into long-run throughput or
learning-quality claims. All development jobs completed and released their NPU
resources.

### Necessary-consumer publication and asynchronous FSDP HF saves (2026-10-26, validation in progress)

The current candidate restores and publishes the dedicated inference group first. The
borrowed group's memory restore and model load run in the existing publication worker;
admission opens only after its version/load ACK. The main loop no longer joins that tail
unless evaluation requires it. Consumers and subsequent state transitions retain their
existing version/readiness fences. Standby training-state staging was already
asynchronous; its source-generation lifetime fence is deliberately retained because its
mmap is reused.

FSDP now supports explicit `saver.mode=async` for full-model HF saves, including
TrainController. Auto keeps its prior behavior, so the fixed baseline remains
synchronous. R030 enables this mode when `AREAL_IDLE_ASYNC_TAIL=1` (default inside the
opted-in idle-training mechanism); set 0 for the prior path. With an available current
DRAM generation, saving avoids gathering live accelerator tensors; generic FSDP uses its
normal full CPU state gather. Rank zero creates independently owned CPU tensors,
including tied aliases and persistent buffers, before returning. A bounded writer
serializes safetensors and copied configuration/tokenizer data. A completion marker
appears only after successful writes; failures propagate to all training ranks. The
marker is not a power-loss durability guarantee.

Only one save per engine can remain pending; the next save joins it. Training-loop
completion joins all final writers, so benchmark time does not hide unfinished output.
The source DRAM map can be reused after CPU staging because the writer owns its copy.
Optimizer/recovery checkpoints, DCP, and LoRA async saves are outside this
implementation and retain synchronous behavior. CPU tests cover generic FSDP staging,
post-staging source mutation, tied parameters, persistent buffers, I/O failure, old-file
rejection, and unchanged auto-mode behavior. Full three-batch NPU correctness and then
six-batch fixed/shared development comparison are pending.

### Asynchronous-tail result (2026-10-26)

The three-batch correctness sequence passed five full state imports. Helper gradient
replay relative L2 was 1.2328% (existing BF16 tolerance 2%). All 339 serialized HF
tensors were read back and exactly matched the staged CPU state. This validates file
contents against the snapshot, not bitwise equivalence between different distributed
training reductions.

The six-batch reused-service comparison completed: fixed 455.835 s, shared 430.845 s,
saving 24.991 s (5.4824%). Reported generated tokens were 358492/353648. The previous
joint-choice pair was 463.256/435.637 s with tokens 356852/365276. The new shared loop
is 4.792 s shorter, but workload differences prohibit attributing that difference to
asynchronous tails. Shared PPO was 245.843 s (previously 250.127 s).

The targeted stages do not yet show a clear net gain. Shared publication totaled 15.481
s versus the previous 15.840 s; B1 improved from 8.835 to 6.977 s, while other
publications varied. Shared save staging/call took 11.550 s and the final join 1.940 s,
totaling 13.490 s versus the previous synchronous save 12.871 s. Inside the worker, CPU
staging was 4.272 s and background writing 2.434 s. Thus substantial time remains
before/around worker staging, and the final-only save offers little subsequent work to
overlap. A cold CPU mode-selection probe took 0.785 s, insufficient to explain the full
gap; no unmeasured cause is claimed.

The final writer was joined before train.end; this comparison does not exclude pending
disk work. All four chart families were generated, with background write and final-write
wait rows added to phase charts. User-facing shortcut:
`/tmp/huawei/areal-benchmarks/20261026-r030-async-tail-comparison`. Evidence:
`../experiments/r030-async-tail-20261026/`. The experiment snapshot used saver mode
auto; a subsequent guard ensures explicit mode sync is respected and does not change
this experiment's path. No claim is made that async HF save alone improves this short
workload.

## same-state asynchronous HF save replay (2026-10-26)

The original 11.549 s foreground call contains a 4.308 s rank-0 engine RPC, including
4.272 s staging. Only 0.036 s is inside that RPC but outside staging. Approximately
7.241 s is outside the engine RPC. Existing logs cannot retrospectively attribute every
millisecond of that outer interval. It is not a seven-second DRAM or optimizer fence.
This corrects the earlier unverified fence hypothesis.

The replay confirms expensive tokenizer metadata handling: per-rank archive/
serialization, tokenizer reconstruction, and writer-side deepcopy. The change captures
save_pretrained files once per save, passes immutable base64 strings through RPC, and
writes them in the background. No tokenizer cache is used: caller mutations are captured
on every save. FSDP async dispatch also avoids importing Archon first. Independent CPU
weight ownership and all state/version fences remain unchanged. Unsafe/reserved
auxiliary paths fail during capture, before weight staging.

## Matched comparison

Same final model state; two CPU/Gloo ranks; production codecs and FSDP save method. Pipe
transport replaces HTTP. No active training, NPU gather, or standby contention is
reproduced. This is a saving-path replay, not a new end-to-end RL measurement. One CPU
thread per worker, unchanged across arms. Order: new, old, old, new.

| Metric (two-run mean)                  |     Old |     New |
| -------------------------------------- | ------: | ------: |
| Foreground, including metadata capture | 6.044 s | 2.298 s |
| Final write join                       | 2.306 s | 2.157 s |
| Total, including file completion       | 8.350 s | 4.455 s |

Foreground reduction: 62.0%; total reduction: 46.7%. Final write waits are included. No
fsync/power-loss durability claim is made.

All four output checkpoints contain 339 tensors exactly equal to the input, and matching
tokenizer vocabulary, special tokens, chat template and sample encodings. 58 focused
tests pass. Source-side optimizer state is not saved by this HF-only path. Timing now
records dispatch, codec, worker queue, individual snapshot fences, DRAM mapping,
independent CPU copy, metadata copy and writer submission. The remaining mandatory CPU
snapshot copy is retained; removing it without pinning an immutable generation would
introduce a race.

Evidence: summary.json, timing-details.json, matched-results.json, correctness.json. No
NPU was used; make-npu-busy was left running. No commit/push.

## full-RL save-path comparison, 2026-10-26

Code RL, FSDP, two shared NPUs, graph inference, 192 questions, 32 questions per batch,
eight answers per question, concurrency 16, scoring concurrency 16, six batches,
offpolicy 1. Initial weights and seed 1 reset before every run. Order: old/new/new/old.
Shared scheduling is identical; four save-related source files are replaced by the
preserved instrumented pre-optimization baseline. Inference engines are reused; training
engines start afresh. Startup is separate.

Before timing, a three-batch DP2/DP1/DP2 correctness run passed five full state imports,
helper gradient relative L2 0.01208594 (existing limit 0.02), and exact readback of 339
saved tensors. All 768 answers were scored without scoring errors. Performance runs
disable full readback/gradient diagnostics as before; each of the four completed 1536
answers and scores without scoring errors.

| Run | Path | RL seconds | Save foreground | Final join | Tokens | Tokens/s |
| --- | ---- | ---------: | --------------: | ---------: | -----: | -------: |
| 1   | old  |    427.202 |          12.158 |      1.927 | 363311 |  850.444 |
| 2   | new  |    422.826 |           3.769 |      1.862 | 358356 |  847.526 |
| 3   | new  |    427.292 |           3.286 |      1.633 | 358476 |  838.949 |
| 4   | old  |    438.011 |          12.562 |      1.956 | 363709 |  830.365 |

Mean foreground saving: 12.360 -> 3.527 s. Final join: 1.942 -> 1.747 s. Combined
critical save time: 14.301 -> 5.275 s, a 9.027 s reduction. Mean RL loop: 432.606 ->
425.059 s, 7.547 s / 1.74% shorter. Mean generated tokens: 363510 -> 358416, 1.40%
fewer. Mean tokens per loop second: 840.404 -> 843.237, only 0.34% higher.

The saving-path benefit repeats, but end-to-end evidence is modest: generation content
differs, earlier PPO/recompute/scoring and entry timings vary, and only two runs per
candidate exist. Do not attribute the entire RL delta to saving. This is a development
warm-session result, not a formal cold-start claim.

The old path's first run directly measured controller serialization 1.085+1.133 s,
worker tokenizer deserialization 3.203/4.277 s, and writer metadata deepcopy 2.966 s.
The two rank timelines overlap. Rank-0 previous-writer collective waited 1.269 s for
rank arrival; the actual snapshot fence was under 1 ms. The new path's first run
captured assets in 0.348 s, used sub-millisecond codec/metadata-copy work, and exposed
0.395/0.642 s DRAM fences that were previously hidden behind metadata work. Independent
CPU weight copy remains required (2.274 s in that new run).

Exact old run timings cannot retrospectively be assigned to the original missing 7.28 s
interval, but the repeated complete pipeline now identifies and removes the expensive
metadata path. No remaining write is excluded from RL duration.

See measurements.json for all stages, input hashes, save events and score metrics.

## latest fixed/shared full RL comparison (2026-10-26)

Same code-RL workload as the old/new saving test: FSDP, two NPUs, TP1 graph, 192
questions, batch32, eight answers/question, concurrency16, scoring16, six batches,
offpolicy1, seed1. Development vLLM reuse with verified initial weight reset. Fixed
retains its original synchronous saving and answer-level scoring; shared uses the latest
policies and optimized async HF saving.

| Metric                                     |   Fixed |  Shared |
| ------------------------------------------ | ------: | ------: |
| RL loop seconds (final file wait included) | 461.807 | 429.981 |
| Generated tokens                           |  361789 |  371932 |
| Tokens / RL loop second                    | 783.421 | 864.996 |
| Save foreground seconds                    |  10.869 |   4.062 |
| Final save join seconds                    |   0.000 |   3.033 |
| Training-engine initialization seconds     | 179.687 | 411.651 |
| Initialization + RL seconds                | 641.494 | 841.632 |

RL loop saving: 31.826 s (6.89%). Token throughput gain: 10.41%. Shared generates more
tokens, but content differs and this is one pair; do not attribute the complete shared
gain to the final saving change. The short run is slower when training-engine
initialization is included. These totals still exclude the earlier reusable inference
service preparation, so are not full job wall times.

Both arms completed all 1536 answers and scores with no code_reward_error events. The
three-batch full correctness gate passed before benchmarking. All six timed checkpoints
across both experiments have matching tokenizer vocabulary, special tokens, chat
template and sample encodings. Performance full readback was disabled as before; the
final write completion is inside train.end.

Four figure families (11 PNG files) are under figures/. All timings, scoring work, stage
arrays and input hashes are in measurements.json. make-npu-busy restored.
