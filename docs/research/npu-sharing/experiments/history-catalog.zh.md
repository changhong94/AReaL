[English](history-catalog.md) | 中文

# 历史实验目录与主要结果

这是跨实验索引，不是同条件排行榜。单位、工作量、回答长度、日志和实现均需按原实验核对。

| 实验                                                                                               | 观测结果                                         | 限制                                 |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------ |
| [npu-share-ab01](evidence/history/npu-share-ab01/comparison.json)                                  | 固定 397.32 / 早期共享 459.42 秒                 | 早期共享更慢；实时采样不同。         |
| [npu-share-fast01](evidence/history/npu-share-fast01/comparison.json)                              | HF 共享 459.42 / 快照 426.58 秒；发布少 33.3%    | 组件与总循环统计范围不同。           |
| [npu-share-hybrid03](evidence/history/npu-share-hybrid03/comparison.json)                          | 同步 426.58 / 非对称 389.89 秒                   | 保存评估及长度存在差异。             |
| [npu-share-background02](evidence/history/npu-share-background02/comparison.json)                  | 此前 389.89 / 后台收尾 408.47 秒                 | 重叠生效，总循环却退化。             |
| [ppo-cache-park02](evidence/history/ppo-cache-park02/comparison.json)                              | PPO 原共享 11.494 / park 10.967 / 常驻 11.096 秒 | 不含映射切换；未证明逐位等价。       |
| [cache-overlap02](evidence/history/cache-overlap02/comparison.json)                                | 同步 100.28 / 提前恢复 94.97 秒                  | 等待被隐藏，输出 token 也减少。      |
| [dynamic-kv-ab01](evidence/history/dynamic-kv-ab01/comparison.json)                                | 固定 82.79 / 动态共享 83.97 秒                   | 不是纯分配器开关消融。               |
| [pause-retain-ab01](evidence/history/pause-retain-ab01/comparison.json)                            | 迁移 167.09 / 保留 164.82 秒                     | 该 RL 未触发 RAM KV 压力。           |
| `hundred-batches01`                                                                                | 固定 2715.66 / 共享 2380.91 秒                   | 100 批，不能直接同六批比较。         |
| [adaptive-drain-comparison01](evidence/history/adaptive-drain-comparison01/comparison.json)        | 立即 159.87 / 5 秒 153.59 / 自适应 158.35 秒     | 迁移更少没有带来最快循环。           |
| [batch-drain-comparison02](evidence/history/batch-drain-comparison02/comparison.json)              | 5 秒 153.93 / 自适应 152.43 / 批次 150.20 秒     | 各策略单次，生成输出不同。           |
| [rolling-drain-comparison01](evidence/history/rolling-drain-comparison01/results-summary.json)     | 旧 150.05 / 三窗口 155.81 秒                     | 退化，同时 token 多约 5.4%。         |
| [refined-drain-comparison01](evidence/history/refined-drain-comparison01/comparison.json)          | 旧 155.52 / 组合修正 145.76 秒                   | 估计器归因看同输入回放。             |
| [no-sibling-comparison01](evidence/history/no-sibling-comparison01/results-summary.json)           | 固定 175.42 / 有同题 154.28 / 无同题 154.98 秒   | 预测改善未令本轮更快。               |
| [queue-rebalance-comparison01](evidence/history/queue-rebalance-comparison01/results-summary.json) | 固定 175.36 / 关闭 152.81 / 开启 151.73 秒       | 再均衡成功数零，没有独立收益证据。   |
| [tp-tail01](evidence/history/tp-tail01/summary.json)                                               | 整组中位数 TP1 16.35 / TP2 19.69 秒              | 纯推理固定输出回放，不是 RL 总循环。 |
| [megatron-comparison02](evidence/history/megatron-comparison02/comparison.json)                    | 固定 159.02 / 共享 147.96 秒                     | c16，与 E001 c32 不同。              |

## 归档清单

39 experiments; 60,613 files; 3,655,122,935 bytes.

原始文件选择性复制到持久目录，逐文件 SHA-256 校验。完整 manifest 保留在归档根目录，排除项也在其中。

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
- `hundred-batches01` (4458 files): 原始材料见归档；无顶层摘要复制。
- `independent-reader01` (1254 files): 原始材料见归档；无顶层摘要复制。
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
- `npu-share-taskaffinity01` (1463 files): 原始材料见归档；无顶层摘要复制。
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
- `pressure-memory02` (1246 files): 原始材料见归档；无顶层摘要复制。
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
- `terminal-tail01` (1137 files): 原始材料见归档；无顶层摘要复制。
- `tp-tail01` (18 files): [summary.json](evidence/history/tp-tail01/summary.json)
- `training-priority02` (1253 files):
  [validation.json](evidence/history/training-priority02/validation.json)
- `withdraw-one01` (1246 files): 原始材料见归档；无顶层摘要复制。

[会话摘录](evidence/conversation-excerpts.json) · [归档摘要](evidence/history/catalog.json) ·
[本机补充分析](evidence/supplementary/manifest.json)
