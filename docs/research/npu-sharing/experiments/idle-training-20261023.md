# Preinitialized single/dual FSDP training experiment

[中文](idle-training-20261023.zh.md)

## Objective and scope

Borrow the dedicated inference NPU for data-parallel training when no runnable
generation remains. Both training engines are initialized before the RL loop, with
independent process groups. Single-card training remains the default. This experimental
opt-in currently supports local two-NPU FSDP, inference TP=1, Qwen-style models, no
auxiliary actor/reference/critic/teacher roles, and no recovery. Training remains eager;
inference retains graph mode.

Set `AREAL_IDLE_TRAINING=1` in the benchmark driver's environment. `run_bench.py`
enables it only in the shared arm; the fixed arm explicitly receives zero. This is not
enabled globally.

## Scheduling and state continuity

At a complete batch boundary, the dispatcher atomically checks that no generation is
running, queued for execution, being dispatched, or newly admissible. Reward-only work
may continue on CPU. Inputs blocked by the off-policy version limit are not runnable
generation. Empty inference does not necessarily mean the entire dataset has been
generated.

The inactive training engine stays in DRAM. On selection, inference admission is fenced
and inference allocations are parked. PyTorch Distributed Checkpoint transfers weights
and Adam state from the original actor to the preinitialized dual-card actor, including
scheduler progress and model version. After the update, state is transferred back to the
original publisher before the existing inference weight-publication path resumes. This
conservative implementation deliberately preserves the established publisher/device
topology. It is not an in-place resize of an FSDP process group.

Every handoff validates version, scheduler epoch, optimizer-step range and a
representative Qwen q-projection weight/moment fingerprint. This is sampled state
verification, not a full-model numerical equivalence proof. Any mismatch fails the
experiment rather than silently falling back to stale weights.

## Timing

Both state transfers, memory role switches, pauses and restoration are inside the
measured RL loop. Extra engine initialization is outside that loop and must be reported
separately. Development vLLM reuse is a measurement tool, not a research speedup.

The single-batch smoke test passed, but was slower: fixed 133.676 s versus shared
325.050 s. Switching in took 105.788 s and switching out 100.512 s. Therefore
preinitializing the engines alone is insufficient: full checkpoint-backed state transfer
dominates the compute savings. Do not claim an end-to-end benefit from this
implementation.

Smoke evidence:
`/root/areal-development/idle-training-20261023/runs/20261023-125018-0700_code_fsdp_c16_b1_7ff34a`.

The six-batch development comparison uses code RL, 32 questions/batch, 8
answers/question, 16 concurrent questions, 16 scoring workers, off-policy limit 1, FSDP,
graph inference, and devices 1 and 2. Its result will be appended after completion.

## Implementation locations

- `areal/trainer/idle_training.py`: preinitialization, selection and state handoffs.
- `areal/trainer/rl_trainer.py`: actor selection and lifecycle integration.
- `areal/engine/fsdp_engine.py`: checkpoint and optimizer/scheduler state continuity.
- `areal/infra/workflow_executor.py`: atomic generation-idle fence.
- `areal/infra/controller/train_controller.py`: controller RPC.
- `benchmark/npu_share/run_bench.py`: shared-arm-only opt-in.
- `tests/test_idle_training.py`: idle-fence and mismatch rejection tests.

Targeted regression suite: 98 tests passed. Single-batch NPU smoke completed with both
state handoffs verified.

## Initialization compatibility fix

The first six-batch attempt completed the fixed arm but failed during dual-actor
initialization: HCCL device port 16666 was already bound. The additional actor now uses
`HCCL_NPU_SOCKET_PORT_RANGE=auto`, matching the existing multiple-inference-process
approach. The failed attempt is retained under
`20261023-132143-0700_code_fsdp_c16_b6_0677c9`; it is not a performance result.

The second attempt completed the first dual-card update and both verified handoffs, then
failed the next single-actor memory budget check. Inactive dual-actor workspace caches
were still mapped. Handoff now finishes gradient lifetime and parks reusable workspace
addresses before offloading weights and moments. Restoring the destination also restores
its retained workspace. The failed run `20261023-135336-0700_code_fsdp_c16_b6_f9d69c` is
retained as compatibility evidence, not a completed comparison.

## Completed six-batch comparison

Workspace:
`/root/areal-development/idle-training-20261023/runs/20261023-143705-0700_code_fsdp_c16_b6_352afc/repetitions/01`.

| Metric                             |   Fixed | Shared with idle training |
| ---------------------------------- | ------: | ------------------------: |
| RL loop, seconds                   | 460.921 |                   866.057 |
| Trainer initialization, seconds    | 180.789 |                   299.103 |
| PPO update total, seconds          | 288.851 |                   263.178 |
| State/memory switch total, seconds |       0 |                   433.920 |
| Reported generated tokens          |  364652 |                    349163 |

Shared was 405.136 seconds slower (87.897%). Batches 1 and 6 used dual-card training;
batches 2–5 kept single-card training because inference was not idle. Four handoffs
preserved sampled weights/moments, scheduler progress and optimizer steps 0→1 and 5→6.
Both arms completed six updates. The lifecycle fixes allowed inference and later
training to continue after a dual-card update.

This is a negative end-to-end result. The checkpoint-backed transfer path dominates the
compute saving. Even initialization was 118.314 seconds longer, outside the reported RL
loop. Generated content differed between arms, so the PPO difference alone is not a
controlled compute speedup; refer to the earlier identical-input replay for that
question. This is one development pair with reusable vLLM, not formal repeated
confirmation. Keep the feature opt-in. A useful next implementation would need efficient
in-memory resharding/state transfer and avoid unnecessary round trips to the single-card
publisher; preinitialization alone cannot remove this overhead.

Compact evidence: [summary](evidence/idle-training-20261023/summary.json),
[phase totals and handoff events](evidence/idle-training-20261023/analysis.json).
External cleanup completed and the previously running utilization container was
restored; container management remains outside project code.
