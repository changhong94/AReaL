English | [中文](R007-single-npu-and-borrowing.zh.md)

# R007: Single-NPU switching and borrowing idle training capacity

Status: Implemented foundation; single-card capacity saving is not a speedup claim.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

Fixed roles leave the training device idle while waiting for rollout. npu_share is one
opt-in boolean combining role switching and memory policy. A single card alternates
inference/training; with two cards, one remains dedicated to inference and the training
card joins the inference pool when safe. Optimizer ownership stays with the training
process. This does not dynamically resize the training distributed group.

## Difficulty and contribution boundary

Coordinate dispatch closure, native pause ACK, request leases, memory admission,
optimizer continuity and version commit. Treat shared-card state as a protocol, not just
a device-placement flag. Initial one-card code did not integrate the two reference
runtimes; slab and mapping integrations arrived later. Existing AReaL asynchronous
rollout is upstream functionality, not a new contribution of this work.

## Evidence and evolution

Single-card smoke preserved optimizer identity through two updates. Pressure tests
advanced Adam through four steps and exercised resident/KV-only/CPU-offload paths. A
16-update comparison took 778.20 s on one shared NPU versus 370.05 s on two dedicated
NPUs: 2.10x elapsed time and about 1.05x reserved card-seconds. Early two-card sharing
was 15.63% slower than fixed roles. Later combined migration sharing completed 100
batches in 2380.91 s versus 2715.66 s (12.33% shorter), with changed policies and
workload; do not attribute that gain to the initial switch alone.

## Next validation

Report wall time, initialization, reserved device-seconds, power and quality separately.
Long-run stability and quality must be evaluated beyond smoke tests; physical compute
occupancy is not equivalent to reserved card-seconds.

## Code locations

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/infra/borrowed_rollout.py](../../../../areal/infra/borrowed_rollout.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

- [areal/api/cli_args.py](../../../../areal/api/cli_args.py)

## Experiment evidence and conversation provenance

- [single-npu-timeshare-smoke03/validation-summary.json](../experiments/evidence/history/single-npu-timeshare-smoke03/validation-summary.json)

- [auto-memory-pressure01/validation-summary.json](../experiments/evidence/history/auto-memory-pressure01/validation-summary.json)

- [npu-share-comparison/comparison.json](../experiments/evidence/history/npu-share-comparison/comparison.json)

- [npu-share-ab01/comparison.json](../experiments/evidence/history/npu-share-ab01/comparison.json)

- `hundred-batches01`: Logs/scripts are in the persistent historical archive.

Conversation excerpt IDs: C010, C013, C014, C015, C091.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
