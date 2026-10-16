English | [中文](R022-multi-npu-rank-budgets.zh.md)

# R022: Multiple inference instances and per-rank TP memory budgets

Status: Recent uncommitted extension; 4-card TP1 and 8-card TP2 functional coverage, no
matched scaling claim.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Mechanism diagram

![R022: Per-rank memory decisions; all-rank readiness for TP execution](R022.png)

2 / 4 / 8 NPUs; one fixed training group. No elastic resizing of training ranks.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R022.svg) | [PNG](R022.png)

## Problem and mechanism

Two-card assumptions hardcode one actor and two engines. Extend the benchmark to 2/4/8
devices, one training group using half the devices, and inference instances whose TP
width divides the inference half. Shared instances borrow training devices. Aggregate
device states without hiding ranks: each card independently reclaims only its own
shortfall; admit a TP group only when every member is ready.

## Difficulty and optimization point

Map controller rank/instance/member to physical devices; coordinate full-weight
publication across training ranks, group ACKs and partial failures. Install scheduler
hooks in EngineCore rather than only device workers. Avoid HCCL port collisions between
training/inference on the same card. Capacity changes must preserve live block IDs.
Sufficient cards retain KV while pressured cards move valid paused KV to RAM, but
collective execution cannot outrun the slowest unready rank.

## Functional evidence

The 8-card TP2 shared validation completed 32 questions x 6 answers / two batches in
79.51 s. First PPO overlapped 44 device windows per borrowed instance, about 2.58 s
each; second PPO had no remaining inference. A separate asymmetric native test reclaimed
4 MiB on one device while retaining the other, then verified byte-exact restoration.
These are separate tests, not proof that that RL run exercised pressure. TP4 has not
received end-to-end validation; no scaling speedup is claimed.

## Next validation

Use matched fixed/shared 4/8-card runs, induce pressure on one TP member, and measure
collective waiting and per-rank memory. This is fixed topology with dynamic
role/admission changes, not elastic training-group resizing.

## Code locations

- [areal/infra/scheduler/local.py](../../../../areal/infra/scheduler/local.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

- [areal/infra/controller/train_controller.py](../../../../areal/infra/controller/train_controller.py)

- [areal/trainer/colocation_memory.py](../../../../areal/trainer/colocation_memory.py)

- [areal/engine/vllm_ext/dynamic_kv.py](../../../../areal/engine/vllm_ext/dynamic_kv.py)

- [areal/engine/vllm_ext/kv_slab.py](../../../../areal/engine/vllm_ext/kv_slab.py)

- [areal/engine/vllm_remote.py](../../../../areal/engine/vllm_remote.py)

## Experiment evidence and conversation provenance

Conversation excerpt IDs: C143.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.

[TP2 validation](../experiments/evidence/multi-npu/validation-summary.json)

[Native asymmetric memory test](../experiments/evidence/multi-npu/native-asymmetric-kv.log)
