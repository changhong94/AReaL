[English](R015-training-cache-parking-prefetch.md) | 中文

# R015: 训练缓存保址解除映射与提前恢复

状态：实验开关已实现；机制已验证，计时回放没有证明严格数值等价。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 诊断与机制

同输入 profiling 加固定组清缓存对照定位到反复分配：第三批首微批分配调用固定 45.4 ms、共享 281.8 ms、固定加清缓存 313.4 ms，算子时间都约
303 ms。为专用 backward 池保留虚拟地址，只解除完全空闲整段的映射，使用前映回。空闲字节没有需保留的数据，不复制到 RAM；有活跃张量的段保持映射。

## 重叠与难点

专用卡继续完成剩余题目时提前恢复，回答和内存都就绪才训练。与前次发布收尾串行，失败唤醒等待者。必须核对段地址、大小及全空闲状态，park 期间禁止清缓存和模型操作。昇腾
autograd CPU 线程未正确继承分配器上下文，因此局部调整 CPU 多线程行为，其成本也要计入。最初 75% 撤回后来改为只剩一题，提前恢复会牺牲一部分共享推理时间。

## 实测结果

39 段约 10.33 GiB 被解除映射，解除/恢复约 0.053/0.279 秒；原共享/park/同池常驻三批 PPO 合计 11.494/10.967/11.096
秒，不含两次切换共 0.664 秒，因此不能仅凭 PPO 认定净收益或逐位等价。提前恢复实验循环 100.28→94.97 秒，收齐后等待 3.64→0.002 秒，但
token 45371→43093。16 题中由第 12 题改到第 15 题撤回后，空等从 4.96/6.72/2.29 降到 0/0/1.95 秒，总循环却在不同实时运行中
82.71→85.88 秒。

## 下一步

按容差比较同输入输出和完整权重，不能只比低精度打印的梯度。分别量化隐藏恢复成本、损失的推理机会和训练分配节省，并按 R004 联合优化就绪时间。

## 代码位置

- [areal/engine/colocation_memory.py](../../../../areal/engine/colocation_memory.py)

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/trainer/training_prefetch.py](../../../../areal/trainer/training_prefetch.py)

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

## 实验证据与会话来源

- [backward-profile01/backward-comparison.json](../experiments/evidence/history/backward-profile01/backward-comparison.json)

- [ppo-cache-park02/comparison.json](../experiments/evidence/history/ppo-cache-park02/comparison.json)

- [ppo-cache-park02/final-comparison.json](../experiments/evidence/history/ppo-cache-park02/final-comparison.json)

- [cache-overlap02/comparison.json](../experiments/evidence/history/cache-overlap02/comparison.json)

- `withdraw-one01`: 日志与脚本见持久历史归档。

会话摘录编号：C054, C058, C060, C087.
[查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。

## 补充的本机分析产物

- [withdraw-one-timeline.json](../experiments/evidence/supplementary/withdraw-one-timeline.json)
