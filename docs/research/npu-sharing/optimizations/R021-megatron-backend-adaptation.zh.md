[English](R021-megatron-backend-adaptation.md) | 中文

# R021: 共享执行的 Megatron 引擎适配

状态：已实现并在记录的小模型配置验证。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

FSDP 专属的分配与发布钩子不会自动适用于 Megatron。使用后端要求的 Python 和运行环境，适配内存池覆盖与训练预算，保持优化器所有权，提供正确 rank
协调的全权重快照/HCCL 发布。先单批验证再做六批对照。

## 难点与贡献边界

后端分配器缓存、分片存储和导入/ABI 兼容会打破 FSDP 假设。启动就绪正确性与计时同样重要。本项是共享协议的后端泛化，不是新 Megatron
训练算法；仍受当前拓扑约束。

## 实测与长尾诊断

c16、96 题每题六答：固定 159.02 秒/138799 token，共享 147.96 秒/137648，实测缩短 6.95%。最后题 Q95 固定 650
token、共享 905；共享 TPOT 还更低，为 42.68 对 47.39 ms，但更长回答吃掉约七秒领先，跨卡执行空档仅约 0.085 秒。后续 E001 c32 为
120.98/109.65 秒，不应把跨实验并发变化当成后端消融。

## 下一步

重复配对种子并回放相同 PPO 输入；扩展模型和分布式布局，独立评估训练后质量，再判断算法等价性或后端通用收益。

## 代码位置

- [areal/engine/megatron_engine.py](../../../../areal/engine/megatron_engine.py)

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/engine/colocation_memory.py](../../../../areal/engine/colocation_memory.py)

- `benchmark/npu_share/run_bench.py`

## 实验证据与会话来源

- [megatron-comparison02/comparison.json](../experiments/evidence/history/megatron-comparison02/comparison.json)

- [megatron-comparison02/summary.json](../experiments/evidence/history/megatron-comparison02/summary.json)

会话摘录编号：C136, C139. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。

上述尚未提交的代码路径记录本次检查时的工作区位置，不包含在研究文档提交中；可在对应实验源码快照中追溯。
