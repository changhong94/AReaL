English | [中文](R002-training-priority-concurrency.zh.md)

# R002: Training-priority inference on the same NPU

Status: Implemented, opt-in; functional overlap verified; isolated speedup remains
workload-dependent.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Mechanism diagram

![R002: Reserve training memory first; admit inference into the remainder](R002.png)

Separate weight copies; memory admission does not guarantee compute priority.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R002.svg) | [PNG](R002.png)

## Problem and mechanism

Strict time slicing leaves inference idle throughout PPO even when memory can
accommodate both processes. Reserve actor state, pending gradients/optimizer state,
training and inference workspaces, and a safety margin first. Admit inference only into
the remaining KV budget, limiting both physical pages and scheduler block IDs. The two
processes retain separate model weights: concurrent execution does not mean modifying
the model currently serving requests.

## Contribution candidate and difficulty

The candidate contribution is coupling a training-memory reservation with live inference
admission rather than merely colocating processes. Admission failure falls back to time
slicing. Publication must close admission and acknowledge pause before changing
inference weights or lifting caps. Memory priority is not compute priority: kernel, CPU
and bandwidth contention can slow PPO. Fixed workspace budgets are not operator-level
demand paging or an OOM guarantee.

## Measured outcome

The code-RL validation reported 211.09 s for prior migration sharing and 200.83 s with
training-priority overlap (about 4.9% shorter). This was not identical generated-answer
replay. The math comparison measured fixed 169.43 s / 134766 tokens and shared 157.84 s
/ 137139 tokens. That 6.84% reduction is a combined-system observation, not this feature
alone. E001 admitted overlap at all six steps yet card 1 produced only 21.3% of tokens:
admission alone does not provide work.

## Next validation

Use matched generated inputs to measure PPO contention and an on/off overlap ablation
with identical KV policy. Report actual intersecting device windows, not just
overlap-begin events, and evaluate latency, reward and memory pressure.

## Code locations

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/trainer/colocation_memory.py](../../../../areal/trainer/colocation_memory.py)

- [areal/engine/vllm_ext/dynamic_kv.py](../../../../areal/engine/vllm_ext/dynamic_kv.py)

- [areal/engine/vllm_ext/kv_slab_adapter.py](../../../../areal/engine/vllm_ext/kv_slab_adapter.py)

## Experiment evidence and conversation provenance

- [training-priority02/validation.json](../experiments/evidence/history/training-priority02/validation.json)

- [math-training-priority01/comparison.json](../experiments/evidence/history/math-training-priority01/comparison.json)

Conversation excerpt IDs: C099.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
