English | [中文](methodology.zh.md)

# Evidence methodology and claim boundaries

## What was reviewed

This review used the current session's original assistant final messages, the visible
user requests, the repository's existing experimental reference document and commit
history, and retained experiment files on host 7.150.15.118. It extracted 47 relevant
historical message excerpts with UTC timestamps and full-message hashes. Conversation
reports are discovery aids, not independent validation. No user credentials or full
session dump were copied into this research directory.

The original session extends beyond NPU sharing into RL quality debugging, E2B/Kuasar
setup and earlier machines. Those are background rather than sharing optimizations. This
review does not claim to have exhaustively revalidated every historical experiment. It
prioritizes the resource-sharing work on 118 and records earlier async experiments as
upstream context (R024).

## Evidence hierarchy

1. **A: Raw traces/configuration/source snapshot.** Best for reconstruction. Inspect
   actual workload, timestamps, version semantics and failures, not only a success flag.
1. **B: Retained machine-readable analysis.** Supports the reported metric, subject to
   its aggregation and any missing source. Most numerical results in these notes are B,
   with selected raw evidence retained alongside them.
1. **C: Conversation/document report.** Useful for intent and chronology. Corroborate
   before promoting it into a paper headline. IDs Cxxx point to extracted messages.
1. **H: Hypothesis.** R001 has motivation evidence, but no implementation or measured
   gain.

No single live comparison establishes a universal speedup. Do not add reductions from
separate configurations or claim a cumulative improvement from chronological runs.

## Comparison contract

- Separate initialization, loop, evaluation/save, final publication and shutdown.
- Match model, exact question hashes, answers per question, token limits, seeds,
  precision, engine, TP/DP, device allocation, cache state and logging controls.
- Match global sandbox concurrency, not per-process semaphore counts.
- `max_concurrent_rollouts` counts question workflows in the math recipe; `max_num_seqs`
  limits native requests per inference instance.
- Fixed execution in this fork is not an untouched upstream baseline. Kernel warmup and
  instrumentation can affect both arms; record their source snapshot.
- Same questions and seed do not guarantee identical answers, consumed batch membership
  or training trajectory. Report generated/replayed tokens and quality. For component
  attribution, replay identical generated tensors and microbatch partitions.
- Repeated paired seeds, randomized or counterbalanced run order and uncertainty
  intervals are still required for strong performance claims.
- Rollout reward/pass count is not independent before/after evaluation. Retain version
  and importance-weight correctness, and run held-out evaluation before claiming
  equivalent learning quality.

## Measurement semantics

Device windows synchronize around execution and include host dispatch/waits. They are
not exclusive per-request kernel time and perturb scheduling. Multiple concurrent
requests may share a window. CPU progress frames are a different timing source.
Two-second hardware sampling can miss peaks; AI Core activity is not achieved FLOPS, and
HBM includes allocated/cached/reserved memory. Token histograms reconstruct single-token
decode-window ends and exclude replay/prefill; count audits matter. Overlap-begin means
permitted overlap; only intersecting execution evidence proves work. Migration
candidates are not actual migrations: count acknowledged route transitions.

## Preservation and provenance

- Full E001 copy:
  `/root/areal-research-artifacts/20261015-144720-0700_math_megatron_c32_b6_d288c2`.
- Selected history: `/root/areal-research-artifacts/history-20261015/`, with SHA-256
  manifest. It excludes model checkpoints and full profiler payloads and is **not** a
  complete runnable backup. Original `/tmp` files remain untouched. Source manifests
  alone cannot recreate missing source files or container layers.
- Small summaries live in [evidence/history](experiments/evidence/history/catalog.json).
- [Conversation excerpts](experiments/evidence/conversation-excerpts.json) retain exact
  historical language; source artifacts may contain Chinese even in the English ledger.
- [Reviewed-code hashes](experiments/evidence/reviewed-code-manifest.json) identify
  today's inspected implementation, not historical measured revisions.
- Current feature commit is `eacd8c4b`; documentation commit `7eccfe95` is the current
  HEAD, with later uncommitted backend/multi-device changes. Do not label all runs as
  experiments of that exact committed tree.

Persistent copies are on the same host, not off-host backups. Before publication, export
approved artifacts to durable project storage and review internal endpoints, paths,
licenses and third-party redistribution rights. This review changes documentation only
and does not rerun training or claim the archived source is universally runnable.
