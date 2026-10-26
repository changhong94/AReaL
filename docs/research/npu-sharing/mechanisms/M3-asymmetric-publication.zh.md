[English](M3-asymmetric-publication.md) | [总览](../core-mechanisms.zh.md)

# M3：非对称异步权重发布

## 解决的问题与主要机制

让已就绪的推理卡尽早恢复，避免共享卡快照和恢复工作阻塞整个训练循环。

稳定权重 → 专用卡 HCCL 与共享快照独立读取 → 逐副本版本确认 → 各自恢复 → 修改权重前汇合。

## 可追溯的实现记录

- [R003](../optimizations/R003-asymmetric-publication.zh.md)
- [R008](../optimizations/R008-flat-weight-snapshots.zh.md)
- [R009](../optimizations/R009-independent-snapshot-reader.zh.md)
- [R010](../optimizations/R010-background-step-tail.zh.md)
- [R011](../optimizations/R011-terminal-publication-elision.zh.md)

这些条目是同一核心机制的组成部分，不代表各自独立创新。原 R 编号和历史证据保持不变。

## 局限、备选与负面结果

R010 早期重叠实验反而变慢；线程分离不消除内存带宽竞争。R011 只在真正终止且评估不需要新权重时消除发布，不以 epoch 边界替代终止判断。

## 需要的独立验证

分别消融扁平快照、独立读取、逐副本恢复和后台收尾；测两个发布分支、训练循环关键路径及版本正确性。

共享整体加速不等于本机制的独立收益或新颖性已确认。参见[当前认可策略与证据](../experiments/2026-10-19-r001-accepted-policy.zh.md)。
