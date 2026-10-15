English | [中文](R015-training-cache-parking-prefetch.zh.md)

# R015: Address-preserving training-cache parking and prefetch

Status: Implemented experimental switches; mechanism verified, exact numerical
equivalence not established by timing replay.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Diagnosis and mechanism

Same-input profiling plus a fixed-mode clear-cache control identified allocation churn:
batch-3 first-microbatch allocation calls 45.4 ms fixed versus 281.8 ms shared and 313.4
ms fixed+clear, while operator time stayed near 303 ms. Preserve a dedicated backward
pool’s virtual addresses, unmap only fully inactive whole segments, then remap before
use. Inactive bytes contain no needed data and are not copied to RAM; live segments
remain mapped.

## Overlap and difficulty

Prefetch restoration while the dedicated engine finishes remaining questions; train only
after both answers and memory are ready. Serialize with the previous publication tail
and fail waiters on error. Verify exact segment address/size/inactive state; no cache
clearing or model operations while parked. Ascend autograd CPU threads did not inherit
the allocator context correctly, requiring a scoped CPU-multithreading change, which
itself needs measurement. Initial 75% withdrawal later became one-question-remaining;
this tradeoff sacrifices some inference time.

## Measured results

Parking 39 segments reclaimed about 10.33 GiB; unmap/remap cost about 0.053/0.279 s. PPO
totals baseline/park/resident were 11.494/10.967/11.096 s, excluding 0.664 s of mapping
transitions; this alone does not prove net gain or bitwise equality. Prefetch
comparison: loop 100.28→94.97 s; post-answer wait 3.64→0.002 s, but output tokens
45371→43093. Withdrawal at 15 rather than 12 of 16 questions reduced idle waits
4.96/6.72/2.29→0/0/1.95 s yet loop regressed 82.71→85.88 s in separate live runs.

## Next validation

Compare same-input outputs and full parameter tensors within tolerance, not only
gradients printed with low precision. Separate hidden restore cost, lost inference
opportunity and training allocation savings; tune readiness jointly as R004 proposes.

## Code locations

- [areal/engine/colocation_memory.py](../../../../areal/engine/colocation_memory.py)

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/trainer/training_prefetch.py](../../../../areal/trainer/training_prefetch.py)

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

## Experiment evidence and conversation provenance

- [backward-profile01/backward-comparison.json](../experiments/evidence/history/backward-profile01/backward-comparison.json)

- [ppo-cache-park02/comparison.json](../experiments/evidence/history/ppo-cache-park02/comparison.json)

- [ppo-cache-park02/final-comparison.json](../experiments/evidence/history/ppo-cache-park02/final-comparison.json)

- [cache-overlap02/comparison.json](../experiments/evidence/history/cache-overlap02/comparison.json)

- `withdraw-one01`: Logs/scripts are in the persistent historical archive.

Conversation excerpt IDs: C054, C058, C060, C087.
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

- [withdraw-one-timeline.json](../experiments/evidence/supplementary/withdraw-one-timeline.json)
