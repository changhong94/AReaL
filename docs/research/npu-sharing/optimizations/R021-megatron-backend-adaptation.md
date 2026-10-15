English | [中文](R021-megatron-backend-adaptation.zh.md)

# R021: Megatron integration for shared execution

Status: Implemented and validated on the recorded small-model configuration.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

FSDP-specific allocation and publication hooks do not automatically apply to Megatron.
Use the backend-compatible Python/runtime, adapt pool coverage and training-state
budgeting, preserve optimizer ownership and provide full-weight snapshot/HCCL
publication with appropriate rank coordination. Validate one batch before six-batch
comparisons.

## Difficulty and contribution boundary

Backend allocator caches, sharded storage and import/ABI compatibility can invalidate
assumptions from FSDP. Startup and readiness correctness matter as much as timing. This
is backend generalization of the sharing protocol, not a new Megatron training
algorithm. Current topology constraints still apply.

## Measured outcome and tail diagnosis

At c16, 96 questions x 6 answers: fixed 159.02 s / 138799 tokens, shared 147.96 s /
137648, a 6.95% observed reduction. The final question Q95 had 650 tokens fixed versus
905 shared; shared TPOT was lower (42.68 vs 47.39 ms) but its longer answer erased
roughly seven seconds of lead. The card transition gap was only about 0.085 s. E001 c32
later measured 120.98/109.65 s; do not treat cross-run concurrency changes as a backend
ablation.

## Next validation

Repeat paired seeds and replay identical PPO inputs. Add broader model and distributed
layouts, and independent post-training quality evaluation before claiming algorithmic
equivalence or general backend benefit.

## Code locations

- [areal/engine/megatron_engine.py](../../../../areal/engine/megatron_engine.py)

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/engine/colocation_memory.py](../../../../areal/engine/colocation_memory.py)

- `benchmark/npu_share/run_bench.py`

## Experiment evidence and conversation provenance

- [megatron-comparison02/comparison.json](../experiments/evidence/history/megatron-comparison02/comparison.json)

- [megatron-comparison02/summary.json](../experiments/evidence/history/megatron-comparison02/summary.json)

Conversation excerpt IDs: C136, C139.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.

Uncommitted code paths above refer to the reviewed working tree and are not included in
this documentation commit; consult the corresponding experiment source snapshots.
