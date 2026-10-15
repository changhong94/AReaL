English | [中文](R013-question-affinity-and-concurrency.zh.md)

# R013: Question affinity and global rollout concurrency

Status: Affinity implemented as an option; higher concurrency helped one workload,
affinity itself did not.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

Training consumes complete question groups. Spreading every answer independently can
leave many partially completed groups; very low per-engine concurrency underutilizes
decode batches. Optional affinity initially places all answers of a question on one
engine using unfinished-group load, while capacity remains request-based. Reassign only
on source closure; completed answers are not rerun. Global question concurrency and
per-engine max_num_seqs are different limits.

## Difficulty and optimization point

Avoid head-of-line blocking when a preferred card is full and track group ownership
beyond diagnostic logs through reward completion. More work in flight increases KV
pressure, staleness and possibly unused work. Affinity is distinct from sibling-length
prediction removed in R019; deleting that estimator did not delete grouping semantics.

## Measured outcome

Three shared runs: request-routing c16 85.22 s; request-routing c32 70.81 s;
question-affinity c32 72.33 s. Initial same-card placement succeeded for 48 consumed
questions, but affinity was 2.1% slower than request routing at equal c32. Earlier
fixed-output inference replay (128 requests x 256 tokens) showed nonmonotonic scaling:
at total c64 one/two replicas took 40.3/27.8 s; at c128, 17.4/19.4 s. More replicas or
concurrency is not uniformly better.

## Next validation

Factorial comparison of affinity, concurrency and engine width with fixed generated
lengths and then live RL. Keep exact question set, off-policy bound and logging
identical.

## Code locations

- [areal/infra/request_router.py](../../../../areal/infra/request_router.py)

- [areal/infra/borrowed_rollout.py](../../../../areal/infra/borrowed_rollout.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

## Experiment evidence and conversation provenance

- `npu-share-taskaffinity01`: Logs/scripts are in the persistent historical archive.

- `inference-fixed01`: Logs/scripts are in the persistent historical archive.

Conversation excerpt IDs: C002, C034, C035.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.

## Supplementary laptop analyses

- [inference-fixed-output-summary.json](../experiments/evidence/supplementary/inference-fixed-output-summary.json)
