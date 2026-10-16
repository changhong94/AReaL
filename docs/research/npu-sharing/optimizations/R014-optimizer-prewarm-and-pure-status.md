English | [中文](R014-optimizer-prewarm-and-pure-status.zh.md)

# R014: Optimizer prewarm and side-effect-free memory status

Status: Implemented; first-step replay improvement observed.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Mechanism diagram

![R014: Prewarm optimizer state; make memory queries observational](R014.png)

Do not change weights, gradients or learning rate during prewarm.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R014.svg) | [PNG](R014.png)

## Problem and mechanism

First Adam use creates pools and large state tensors on the critical path; memory-status
queries previously cleared caches. Asynchronously prepare the pool/state before training
when budgets permit, with step still zero and no parameter/LR/gradient update. Make
status observational: keep coverage/budget checks but remove unconditional
empty_cache/device synchronization.

## Difficulty and optimization point

If memory is insufficient, create only the pool. Publish initialized state only after
stream completion, and join before training, save/load, offload and destruction. Retain
allocator callback lifetimes. This removes avoidable cold allocation and observer side
effects, not optimizer mathematics.

## Measured outcome

Identical-input replay initially showed PPO 4.894/3.726/3.662 s shared versus
4.394/3.473/3.408 fixed. After prewarm/pure-status changes, shared was 4.259/3.702/3.671
s, about 5.3% less in total, mainly batch 1. Batches 2/3 still had backward overhead,
leading to R015. Later budgeted cache retention did not eliminate that residual by
itself.

## Next validation

Split pool creation, Adam allocation, zeroing and synchronization. Check
weight/optimizer continuity and account for initialization shifted earlier, not
eliminated.

## Code locations

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/engine/colocation_memory.py](../../../../areal/engine/colocation_memory.py)

- [areal/trainer/colocation_memory.py](../../../../areal/trainer/colocation_memory.py)

## Experiment evidence and conversation provenance

- [ppo-replay01/comparison.json](../experiments/evidence/history/ppo-replay01/comparison.json)

- [ppo-replay01/weight-comparison.json](../experiments/evidence/history/ppo-replay01/weight-comparison.json)

- [ppo-prewarm02/comparison.json](../experiments/evidence/history/ppo-prewarm02/comparison.json)

- [ppo-cache-budget01/comparison.json](../experiments/evidence/history/ppo-cache-budget01/comparison.json)

Conversation excerpt IDs: C049, C052, C055.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
