# R001 live KV scheduling validation — 2026-10-16

[中文](2026-10-16-r001-live-kv-scheduling.zh.md)

## Outcome

A live TP=1 KV handoff path and an opt-in cost-aware scheduler were implemented. The
six-batch arithmetic comparisons did **not establish additional speedup from
running-request KV migration**. Both concurrency levels selected zero actual KV
handoffs. This feature remains experimental and disabled by default.

## Workload and controls

- Qwen2.5-1.5B-Instruct, FSDP, two Ascend NPUs.
- 96 GSM8K questions, six answers per question, 16 questions per training batch, six
  batches, one epoch; max generated tokens 1024; off-policy head limit 2.
- Question concurrency 16 or 32; native per-engine sequence capacity is
  `max(64, concurrency * 4)`, hence 64 or 128 respectively.
- Within each matched pair, identical memory policy, transport initialization, planner,
  telemetry, model, data, and configuration. The internal `AREAL_KV_REBALANCE_EXECUTE`
  override is 0 for the no-execution control and 1 for the enabled policy.
  Queued-request scheduling remains active in both.
- Times below measure the training loop, excluding initialization and shutdown.

| Question concurrency | Policy execution | Loop seconds | Answers | Output tokens | KV migrations |
| -------------------- | ---------------- | -----------: | ------: | ------------: | ------------: |
| 16                   | Disabled control |      162.360 |     576 |        135945 |             0 |
| 16                   | Enabled          |      150.860 |     576 |        136471 |             0 |
| 32                   | Disabled control |      124.868 |     576 |        133891 |             0 |
| 32                   | Enabled          |      124.524 |     576 |        134361 |             0 |

Since there were no KV migrations, these differences cannot be attributed to KV
migration. The c16 difference also demonstrates substantial short-run tail and
scheduling variability. The c32 difference is approximately 0.34 seconds. One pair per
setting does not prove statistical non-inferiority or zero regressions in other
workloads.

The c16 pair used the same pre-reconciliation planner snapshot. Three recoveries were
rejected due to admission/completion snapshot races. The corrected planner was replayed
against all five saved snapshots and still selected no transfers. The c32 pair ran the
corrected planner online; matching source hashes are retained in the evidence. These
revision boundaries must be preserved in later analysis.

## Why additional work did not imply useful migration

When router waiters exist, newly available destination capacity serves those waiters
first. Otherwise, the gate evaluates completion-time savings after transport, ownership,
planning, and conservative worker-stall costs. In the observed context bucket 512,
dedicated-engine mean decode-frame intervals were 50.854 ms, 53.283 ms, and 55.846 ms in
concurrency buckets 32, 64, and 128. Splitting requests does not divide autoregressive
decode time in proportion to request counts. These are observational pooled samples, not
a controlled concurrency microbenchmark.

An offline sensitivity check raised the move-count cap from 16 to 64 and 128 on the
saved c32 snapshots. It still selected no moves. This excludes the cap alone as the
cause of these particular rejected decisions; it is not evidence about unmeasured
workloads.

## Functional evidence and safeguards

The actual vLLM continuation probe imported 91 computed KV tokens (3.5 MiB) after 34
generated tokens. All 256 final greedy output tokens matched the uninterrupted baseline,
with maximum log-probability difference 0. The measured source packing was 11.568 ms and
source-side HCCL was 5.073 ms. The interval from the first freeze response event to
destination import was approximately 125 ms; this is not just wire-transfer time and is
not a complete coordinator latency measurement.

The scheduler preserves version/lease fencing, native token/block validation,
destination continuation capacity rounded per KV block, explicit import acknowledgment,
and rollback before source abort if training becomes ready. Balanced pairs and
unprofitable plans do not migrate requests. Ambiguous failures fail closed. This reduces
risk but is not a universal no-regression guarantee.

The latest targeted regression suite passed **111 tests**. A separate forced
single-request RL integration diagnostic also passed. It is excluded from all
performance claims, because it bypasses the profitability gate.

## Evidence

- [Matched controls, paths, settings, and source hashes](evidence/r001-kv-scheduling-20261016/matched-controls.json)
- [Initial fixed/shared diagnostics](evidence/r001-kv-scheduling-20261016/initial-comparison.json)
- [Observed decode service](evidence/r001-kv-scheduling-20261016/decode-service.json)
- [Candidate-count sensitivity](evidence/r001-kv-scheduling-20261016/candidate-limit-sensitivity.json)
- [Regression test log](evidence/r001-kv-scheduling-20261016/unit-tests.log)
- [Deterministic live KV correctness](evidence/r001-kv-live-20261016/correctness.json)

Full logs and frozen source remain in the persistent benchmark workspaces listed in the
JSON evidence. No performance charts were added to this report.

## Full RL routing integration result

The isolated diagnostic completed two batches and all 192 answers successfully. It moved
one running request through the ordinary RL client/router, imported 331 computed KV
tokens (10.5 MiB), and observed `import_applied` at the destination. The coordinator
interval from planning to confirmation was 0.272716 seconds, excluding the preceding
snapshot collection. Packing was 15.250 ms; source-side HCCL was 1.550 ms and
destination-side HCCL including peer wait was 44.138 ms. These overlapping communication
measurements must not be added together.

The gain override existed only in that workspace's frozen source copy; the repository
profitability gate was not bypassed.
[The integration evidence and explicit source override](evidence/r001-kv-scheduling-20261016/rl-integration.json)
retain its scope and hashes. The process exited with code 0.

Following this measurement, the repository initial per-request handoff cost was raised
from 0.1 to 0.3 seconds; later observed costs can raise it further. The matched online
comparisons above used the earlier 0.1-second seed. A higher cost is a conservative
follow-up, not a new measured speedup. The measured 0.273 seconds also includes
coordination and planning; it must not be described as wire latency.

The final 0.3-second cost seed was also replayed against all ten saved c16/c32 recovery
snapshots: all still selected zero moves.
[Replay evidence](evidence/r001-kv-scheduling-20261016/cost-seed-replay.json). The
cost-seed update passed 15 targeted coordinator/planner tests.
