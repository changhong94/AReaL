English | [中文](history-catalog.zh.md)

# Historical experiments and key results

This is a cross-experiment index, not a matched leaderboard. Check scope, workload,
lengths, logging and source for each run.

| Experiment                                                                                         | Observation                                                | Limitation                                                  |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| [npu-share-ab01](evidence/history/npu-share-ab01/comparison.json)                                  | Fixed 397.32 / initial sharing 459.42 s                    | Initial sharing slower; live samples differ.                |
| [npu-share-fast01](evidence/history/npu-share-fast01/comparison.json)                              | HF sharing 459.42 / snapshots 426.58 s; publication -33.3% | Component and loop scopes differ.                           |
| [npu-share-hybrid03](evidence/history/npu-share-hybrid03/comparison.json)                          | Synchronous 426.58 / asymmetric 389.89 s                   | Save/evaluation and lengths differ.                         |
| [npu-share-background02](evidence/history/npu-share-background02/comparison.json)                  | Previous 389.89 / background tail 408.47 s                 | Overlap worked but loop regressed.                          |
| [ppo-cache-park02](evidence/history/ppo-cache-park02/comparison.json)                              | PPO baseline 11.494 / park 10.967 / resident 11.096 s      | Excludes mapping transitions; no bitwise-equivalence claim. |
| [cache-overlap02](evidence/history/cache-overlap02/comparison.json)                                | Synchronous 100.28 / prefetch 94.97 s                      | Wait hidden; token output also fell.                        |
| [dynamic-kv-ab01](evidence/history/dynamic-kv-ab01/comparison.json)                                | Fixed 82.79 / dynamic sharing 83.97 s                      | Not a pure allocator on/off ablation.                       |
| [pause-retain-ab01](evidence/history/pause-retain-ab01/comparison.json)                            | Migration 167.09 / retained 164.82 s                       | No RAM KV pressure in that RL run.                          |
| `hundred-batches01`                                                                                | Fixed 2715.66 / sharing 2380.91 s                          | 100 batches; do not compare directly with six-batch runs.   |
| [adaptive-drain-comparison01](evidence/history/adaptive-drain-comparison01/comparison.json)        | Immediate 159.87 / 5 s drain 153.59 / adaptive 158.35 s    | Fewer migrations did not yield the fastest loop.            |
| [batch-drain-comparison02](evidence/history/batch-drain-comparison02/comparison.json)              | 5 s 153.93 / adaptive 152.43 / batch 150.20 s              | One pair per policy with different outputs.                 |
| [rolling-drain-comparison01](evidence/history/rolling-drain-comparison01/results-summary.json)     | Previous 150.05 / three-window 155.81 s                    | Regression, with 5.4% more tokens.                          |
| [refined-drain-comparison01](evidence/history/refined-drain-comparison01/comparison.json)          | Previous 155.52 / refinements 145.76 s                     | Use same-input replay for estimator attribution.            |
| [no-sibling-comparison01](evidence/history/no-sibling-comparison01/results-summary.json)           | Fixed 175.42 / with sibling 154.28 / without 154.98 s      | Better prediction did not make this loop faster.            |
| [queue-rebalance-comparison01](evidence/history/queue-rebalance-comparison01/results-summary.json) | Fixed 175.36 / disabled 152.81 / enabled 151.73 s          | Zero successful rebalances; no isolated benefit.            |
| [tp-tail01](evidence/history/tp-tail01/summary.json)                                               | TP1 16.35 / TP2 19.69 s, group median                      | Inference-only fixed-output replay, not a full RL loop.     |
| [megatron-comparison02](evidence/history/megatron-comparison02/comparison.json)                    | Fixed 159.02 / sharing 147.96 s                            | c16; distinct from E001 c32.                                |

## Archive inventory

39 experiments; 60,613 files; 3,655,122,935 bytes.

Selected raw files were copied to persistent storage with per-file SHA-256 verification.
The full manifest and omission list remain at the archive root.

`/root/areal-research-artifacts/history-20261015`

- `adaptive-drain-comparison01` (1625 files):
  [comparison.json](evidence/history/adaptive-drain-comparison01/comparison.json);
  [results-summary.json](evidence/history/adaptive-drain-comparison01/results-summary.json)
- `auto-memory-pressure01` (45 files):
  [validation-summary.json](evidence/history/auto-memory-pressure01/validation-summary.json)
- `backward-profile01` (2630 files):
  [backward-comparison.json](evidence/history/backward-profile01/backward-comparison.json)
- `batch-drain-comparison02` (1627 files):
  [comparison.json](evidence/history/batch-drain-comparison02/comparison.json);
  [results-summary.json](evidence/history/batch-drain-comparison02/results-summary.json)
- `bounded-drain-comparison01` (1612 files):
  [comparison.json](evidence/history/bounded-drain-comparison01/comparison.json)
- `cache-overlap02` (2503 files):
  [comparison.json](evidence/history/cache-overlap02/comparison.json)
- `code-three-batches02` (1362 files):
  [comparison.json](evidence/history/code-three-batches02/comparison.json)
- `cold-start-diag02` (1337 files):
  [comparison.json](evidence/history/cold-start-diag02/comparison.json)
- `dynamic-kv-ab01` (1351 files):
  [comparison.json](evidence/history/dynamic-kv-ab01/comparison.json)
- `dynamic-kv03` (1246 files):
  [summary.json](evidence/history/dynamic-kv03/summary.json)
- `hundred-batches01` (4458 files): Raw archive only; no top-level summary copied.
- `independent-reader01` (1254 files): Raw archive only; no top-level summary copied.
- `kernel-warmup02` (1426 files):
  [comparison.json](evidence/history/kernel-warmup02/comparison.json)
- `kv-slab-smoke03` (35 files):
  [validation-summary.json](evidence/history/kv-slab-smoke03/validation-summary.json)
- `math-training-priority01` (1456 files):
  [comparison.json](evidence/history/math-training-priority01/comparison.json)
- `megatron-comparison02` (1507 files):
  [comparison.json](evidence/history/megatron-comparison02/comparison.json);
  [summary.json](evidence/history/megatron-comparison02/summary.json)
- `no-sibling-comparison01` (2781 files):
  [results-summary.json](evidence/history/no-sibling-comparison01/results-summary.json)
- `npu-share-ab01` (1780 files):
  [comparison.json](evidence/history/npu-share-ab01/comparison.json)
- `npu-share-background02` (1465 files):
  [comparison.json](evidence/history/npu-share-background02/comparison.json)
- `npu-share-comparison` (5 files):
  [comparison.json](evidence/history/npu-share-comparison/comparison.json)
- `npu-share-fast01` (1550 files):
  [comparison.json](evidence/history/npu-share-fast01/comparison.json)
- `npu-share-hybrid03` (1469 files):
  [comparison.json](evidence/history/npu-share-hybrid03/comparison.json)
- `npu-share-taskaffinity01` (1463 files): Raw archive only; no top-level summary
  copied.
- `pause-retain-ab01` (1483 files):
  [comparison.json](evidence/history/pause-retain-ab01/comparison.json)
- `post-warmup-ab01` (1344 files):
  [comparison.json](evidence/history/post-warmup-ab01/comparison.json)
- `ppo-cache-budget01` (2489 files):
  [comparison.json](evidence/history/ppo-cache-budget01/comparison.json)
- `ppo-cache-park02` (3639 files):
  [comparison.json](evidence/history/ppo-cache-park02/comparison.json);
  [final-comparison.json](evidence/history/ppo-cache-park02/final-comparison.json)
- `ppo-prewarm02` (2391 files):
  [comparison.json](evidence/history/ppo-prewarm02/comparison.json)
- `ppo-replay01` (1460 files):
  [comparison.json](evidence/history/ppo-replay01/comparison.json);
  [weight-comparison.json](evidence/history/ppo-replay01/weight-comparison.json)
- `pressure-memory02` (1246 files): Raw archive only; no top-level summary copied.
- `queue-rebalance-comparison01` (1631 files):
  [results-summary.json](evidence/history/queue-rebalance-comparison01/results-summary.json)
- `refined-drain-comparison01` (2627 files):
  [comparison.json](evidence/history/refined-drain-comparison01/comparison.json);
  [results-summary.json](evidence/history/refined-drain-comparison01/results-summary.json)
- `rolling-drain-comparison01` (2620 files):
  [results-summary.json](evidence/history/rolling-drain-comparison01/results-summary.json)
- `single-npu-timeshare-smoke03` (36 files):
  [validation-summary.json](evidence/history/single-npu-timeshare-smoke03/validation-summary.json)
- `tail-prediction01` (6 files):
  [summary.json](evidence/history/tail-prediction01/summary.json)
- `terminal-tail01` (1137 files): Raw archive only; no top-level summary copied.
- `tp-tail01` (18 files): [summary.json](evidence/history/tp-tail01/summary.json)
- `training-priority02` (1253 files):
  [validation.json](evidence/history/training-priority02/validation.json)
- `withdraw-one01` (1246 files): Raw archive only; no top-level summary copied.

[Conversation excerpts](evidence/conversation-excerpts.json) ·
[Archive metadata](evidence/history/catalog.json) ·
[Supplementary local analyses](evidence/supplementary/manifest.json)
