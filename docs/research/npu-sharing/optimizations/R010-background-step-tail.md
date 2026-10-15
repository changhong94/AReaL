English | [中文](R010-background-step-tail.zh.md)

# R010: Move shared publication and step-tail work off the main loop

Status: Implemented; overlap verified, historical first comparison regressed.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

Even after the dedicated engine resumes, synchronous snapshot export, shared
restoration, logging and cleanup can stop the trainer entering prepare_batch. Queue the
shared tail onto a serial executor and let the main loop acquire the next batch after
dedicated readiness. Join the tail before the next actor mutation, evaluation or
shutdown; perform ordering-sensitive data cleanup before publication.

## Difficulty and optimization point

Expose parallel work without races in statistics, snapshots or worker destruction.
Background exceptions must wake batch waiters rather than hang. Overlapping stage
durations cannot be summed as wall time. Earlier entry into prepare_batch is a mechanism
success, not necessarily a faster loop.

## Measured outcome

All 15 ordinary transitions overlapped publication. First dedicated readiness 39.6149 s,
next acquisition 39.6160 s, shared readiness 43.9598 s. Nevertheless final
publication/loop were 377.51/408.47 s versus previous 356.99/389.89 s: about 5.75%/4.76%
slower. Keep this negative result rather than retroactively describing all backgrounding
as a speedup.

## Next validation

Separate exposed wait from useful overlap; use identical workloads and include
evaluation and terminal fences. This infrastructure enables R015 prefetch but is not its
isolated result.

## Code locations

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/trainer/training_prefetch.py](../../../../areal/trainer/training_prefetch.py)

## Experiment evidence and conversation provenance

- [npu-share-background02/comparison.json](../experiments/evidence/history/npu-share-background02/comparison.json)

Conversation excerpt IDs: C024, C025.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
