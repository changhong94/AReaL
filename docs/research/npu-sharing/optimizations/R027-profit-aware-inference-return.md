English | [中文](R027-profit-aware-inference-return.zh.md)

# R027: Profit-aware return of training devices to inference

Status: Opt-in prototype implemented; development validation in progress. No performance
benefit established. Recorded: 2026-10-20 (America/Vancouver). Primary mechanism: M2
scheduling. M1 provides memory feasibility; M3 provides publication and version
readiness.

## Problem

Returning a training device to inference after every update can incur snapshot export,
weight loading, memory transitions and a later training restore, yet provide only a
short useful inference interval. Higher inference occupancy alone does not establish a
shorter training loop.

## Proposed decision

At each training completion, compare predicted next-batch readiness and training
completion under two actions: keep the shared device ready for training, or return it to
inference. Use the last three completed batches, or all available completed batches
during warm-up. Estimate remaining question-group work from observed progress and
output-length distributions, never from future final lengths. Include all answers
required by a question and account for requests already queued, executing or eligible
under the off-policy limit.

Define gain as predicted critical-path completion time without switching minus predicted
completion time with switching. The latter includes only unhidden publication, restore
and interference costs, along with reduced inference service time. Switch when gain
exceeds a configurable uncertainty margin; log both counterfactual estimates, input
sample counts and the decision. Avoid counting an overlapped cost twice. Missing or
stale observations must use an explicit conservative fallback and must not silently
disable shared inference forever.

Throughput estimates must be conditioned on concurrency, engine availability and
training overlap. The additional replica's isolated throughput cannot simply be added to
dedicated-engine throughput. Preserve version, memory-budget and training-priority
constraints. A skipped return must not leave stale model weights eligible for future
requests; update before reopening admission.

## Prerequisites and code locations

Correct incremental memory accounting in `areal/trainer/colocation_memory.py` and
per-transition calibration timing in `areal/infra/controller/rollout_controller.py`
first. Candidate integration points are `areal/trainer/rl_trainer.py`, rolling drain
calibration, request routing and publication completion. Implementation must define the
lifecycle of skipped snapshots and pending weight versions before bypassing any work.

## Validation

Compare always-return and profit-aware return on identical conditions: graph inference,
FSDP, two NPUs, 96 questions, six answers per question, concurrency 16, six batches,
off-policy 2. Keep budget/timing fixes identical between arms. Include workloads with
ample inference work, training-bound workloads, last-batch tails and insufficient
history. Report loop time, batch readiness, tokens, useful shared inference intervals,
switch costs, skipped returns and prediction error. Warm engine reuse is a development
tool; final confirmation uses independent engine startup. No performance benefit is
claimed yet.

## Motivating measurement

[Budget/timing correction experiment](../experiments/2026-10-20-incremental-budget-timing.md):
both warm-session shared runs admitted inference in all six batches, yet were slower
than fixed. Approximately 20 seconds per run waited for the previous publication tail.
This supports studying the decision, not a claim that R027 is effective.

## Prototype implementation (2026-10-20)

`AREAL_PROFIT_AWARE_RETURN=1` enables the decision for the asymmetric shared pool. It is
off by default while under evaluation. `AREAL_RETURN_MARGIN_SECONDS` is the required
predicted saving (default zero).

The controller freezes causal observations at training completion.
`areal/infra/profit_return.py` uses the rolling three-batch length/service calibration,
current complete question groups, and up to three measured return/restore costs. It
compares fixed-only group readiness with readiness after delayed additional capacity;
existing running fixed requests keep their placement. Missing metrics cause a normal
return to collect observations. Already-complete next batches do not need a return. Each
decision is recorded as `inference_return_decision`.

The trainer decides before fixed HCCL publication. A keep decision omits independent
shared snapshot creation, memory switching, shared model loading and rebalance. The
fixed engine still receives the new version. A separate held state blocks generic resume
and training-time inference on stale borrowed weights. A later successful full load
clears the hold. Evaluation forces normal publication; terminal handling is unchanged.
Preserved-request pause mode now evacuates outstanding requests before keeping the
device resident for training; it no longer forces a return to inference.

The initial estimator uses conditional mean residuals, not an expectation over the
maximum of multiple sampled answers, and charges restoration after predicted data
readiness. These are approximations, not validated group-tail or overlap predictions.
Record failures and prediction error; do not infer a benefit from increased occupancy or
from a different generated token count.

Development experiment root: `/root/areal-development/r027-profit-return-20261020/`. Two
warm-session fixed/shared pairs use 96 questions, six answers/question, concurrency 16,
six batches, FSDP, graph, offpolicy 2. This is distinct from the new benchmark defaults
of 32 questions/batch and eight answers/question. Final performance confirmation must
use independent engine startup.

## Accepted rule: mandatory evacuation after a keep-training decision

Decision recorded on 2026-10-20 (America/Vancouver), following explicit user direction.
This supersedes the earlier proposal to reapply migration profitability tests and the
prototype's unconditional return fallback for retained requests.

Separate two decisions:

1. R027 decides whether returning the training device to inference is worthwhile.
1. Once the action is **keep training resident**, **every paused request on that device
   must leave**. Do not apply R001's positive-gain, tail-length, probe-count,
   transaction-time, or balanced-load filters to this mandatory evacuation. Do not leave
   expensive requests behind or reverse the keep decision merely because moving them is
   expensive.

Transport is version-aware:

| Condition                                                                | Required behavior                                                                                                                                             |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Matching weight version and compatible KV layout, resident/exportable KV | Transfer request state and KV, then continue on a fixed inference replica.                                                                                    |
| Different weight versions                                                | Do not transfer/import KV. Transfer prompt, generated token prefix and remaining generation state; rebuild KV by prefill under target weights, then continue. |
| Same version but KV cannot safely be exported/imported                   | Use token-prefix replay rather than leave the request paused indefinitely; record the technical fallback.                                                     |
| Target is busy or lacks immediate capacity                               | Queue/backpressure on fixed inference capacity. This is a feasibility constraint, not an economic veto.                                                       |
| Transfer outcome is ambiguous                                            | Preserve single-owner correctness and surface/reconcile the failure; never silently drop, duplicate, or claim completion.                                     |

Cross-version continuation does not recompute earlier generated tokens. Preserve earlier
token/log-prob/version metadata and label newly generated tokens with the actual target
version; retain existing off-policy checks. It is not equivalent to generating the
entire answer from the new version. No KV reuse across a weight mismatch is allowed.

The implementation must close source admission, fence in-flight execution and record the
complete evacuation set. Source execution ownership must not be released until the
staged transfer or prefix continuation has an acknowledged, recoverable handoff. KV
staging alone is not proof of import; log staging, source abort, target continuation and
`import_applied` separately. Coordinate publication ordering and clear obsolete KV
tickets when versions advance.

### Why this rule differs from optional R001 rebalance

R001 optionally borrows work for load balancing and may reject a costly transfer. R027
evacuation is a liveness obligation after withdrawing a device: leaving paused work
there can prevent a question group and its batch from ever completing. Thus completion
and ownership come first; migration costs are measured for analysis, not used as a
second admission threshold.

### Evidence and implementation status

The mandatory evacuation code now supports same-version paused KV export and
cross-version prefix replay. Direct export copies each request block from its actual
backing: resident HBM or pinned DRAM backup. It does not restore the whole pool, resume
source inference, change source virtual addresses, or discard backups during staging.
Reverse borrowed-to-fixed HCCL channels are initialized for retained-request mode.
Runtime validation is tracked separately from implementation; the earlier frozen R027
performance comparison did not include this path.

Validation must cover all-paused evacuation, mixed weight versions, full target queues,
multi-answer group completion, same-version KV import, prefix replay after a version
mismatch, publication races, partial transport failure, and no stranded/duplicate
request. Report evacuated/remaining counts, transport type, source/target versions,
bytes, replay tokens, latency and final batch completion. Keep no-benefit and failure
cases for future paper discussion.

### Paused-export protocol and limits (2026-10-20)

1. Close source admission, wait for an acknowledged native preserve-pause, and enumerate
   all source leases, including requests waiting at the HTTP gate.
1. Match each source lease's actual weight version against ready fixed replicas. Global
   trainer version is not a substitute. Lock the destination epoch before exporting;
   retain that fence until native import/replay acknowledgement, not merely route
   acquisition.
1. For an exportable same-version request, gather blocks in tensor/block order from
   mapped HBM or pinned host page slices into a bounded HBM staging buffer. No unmapped
   tensor view is read. Stage all TP shards before any enters HCCL; a rejection on one
   shard prevents collective entry by all shards.
1. Transfer and register the target ticket, then abort only the exact paused source
   attempt. Source lease release follows its abort response. A full target queues the
   continuation. A publication-close operation must still admit its already-pinned
   continuation; otherwise close and import would deadlock.
1. Observe `import_applied` or the native replay fallback before advancing to another
   staged request. After all direct transfers, abort/replay all remaining requests and
   HTTP-gate waiters. Only then discard preserved source pages/backups; keep source
   inference admission closed.

The existing safety bound is 64 MiB of staged payload **per TP member**, with staging
headroom reserved. This is not an economic veto: oversized, unstable/preempted,
unsupported-sampling, or memory-infeasible direct exports fall back to prefix replay.
Chunked large-request KV transport is not implemented. Ambiguous HCCL/abort failures
stop the operation rather than risk duplicate execution. Equal TP, compatible
full-attention layout and PP=1 remain required; this does not implement R026 cross-TP
migration.

Cross-version request migration is supported through prefix replay, **not cross-version
KV reuse**. KV encodes intermediate activations computed with a particular set of
weights. Unchanged tensor shapes do not establish semantic compatibility. Preserve old
token logprobs and version labels; the new suffix uses the target model and existing
off-policy semantics. If single-version whole-answer semantics are required, restart the
answer or retain the old model instead—neither is silently substituted here.

Implementation: `dynamic_kv.py` (mixed-backing copy), `kv_handoff.py` (paused-source
packing and TP preflight), `areal_vllm_server.py` (guarded pause/abort endpoints),
`request_router.py` (mandatory routing and epoch fences), `kv_rebalance.py` (serialized
mandatory handoffs), and controller initialization/evacuation. This protocol is a
correctness/liveness extension; no speedup is claimed without a corresponding
experiment.

Unit evidence includes byte-exact mixed HBM/DRAM packing without remapping,
cross-version replay, full-target queues, previously migrated requests, epoch
publication fences, mixed-transport evacuation and TP preflight rejection. Device
evidence is stored under `/root/areal-development/retained-kv-20261020/`; consult its
result files before claiming live validation.

### Device validation result

The graph-mode TP=1 paused-source probe completed with native `import_applied`: all 256
output tokens matched uninterrupted deterministic generation, and maximum logprob error
was 0.0. A separate NPU test copied 3 MiB from mixed HBM/DRAM backing byte-for-byte,
kept the source paused, and remapped zero pages. The regression suite passed 128 tests.
[Evidence](../experiments/evidence/r027-retained-20261020/summary.json).

These are separate memory and request-continuation checks, not a full RL performance
result. A follow-up graph-mode TP=2 probe on 2026-10-21 also confirmed `import_applied`,
256 identical output tokens and zero maximum logprob error.
[TP=2 evidence](../experiments/evidence/r027-retained-tp2-20261021/result.json). This
validates one same-TP pair, not arbitrary replica counts or multihost transport.
Cross-version routing has unit coverage; this evidence does not establish cross-version
KV reuse (which remains prohibited).

## Resident training budget experiment (2026-10-21)

Skipping an inference return should also avoid unnecessary allocator churn before the
next training batch. The resident fast path still checks the memory budget. An
experimental `AREAL_RESIDENT_BUDGET=1` path credits only verified reusable backward
storage and fully inactive, non-expandable default-pool segments on the current device
and stream. Default-pool credit is restricted to single-rank training until
communication-stream reuse is validated. Optimizer growth, weight publication,
restoration and the safety reserve remain explicit requirements. Parked or offloaded
actor storage disables this shortcut; an insufficient budget falls back to the existing
reclamation path.

This is not a claim that all reserved allocator memory is reusable. Unknown snapshot
fields, foreign streams, live allocations, private pools and partially inactive segments
receive no default-pool credit. A separate physical device-budget check prevents
allocator credit from concealing an exceeded budget. The option remains experimental
until performance and correctness measurements are complete. Checkpoint saving remains
enabled in both arms; added tracing separates CPU state gathering, dtype conversion and
serialization without changing checkpoint contents.

Development and formal results are recorded in
[the resident-training experiment](../experiments/2026-10-21-r027-resident-training.md),
including unsuccessful variants.

## Default enablement (2026-10-21)

Following two positive independent-startup comparisons, profit-aware return and
resident-cache budgeting are enabled by default. Benchmark and coverage shared arms
explicitly enable both; fixed arms explicitly disable both. Existing safety checks and
single-rank default-pool credit restrictions remain. Explicit runtime environment values
of `0` disable either mechanism. Historical experiment statements about opt-in status
describe their configuration at the time.
