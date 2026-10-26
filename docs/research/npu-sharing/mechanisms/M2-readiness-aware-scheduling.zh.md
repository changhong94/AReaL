[English](M2-readiness-aware-scheduling.md) | [总览](../core-mechanisms.zh.md)

# M2：面向训练就绪时间的请求调度

## 解决的问题与主要机制

预测剩余工作，在排空、排队搬移和运行中迁移之间选择，尽早满足训练启动条件。

最多三批统计窗口 → 剩余时间预测 → 比较数据与训练卡就绪时间 → 成本感知调度 → 反馈实际成本。

## 可追溯的实现记录

- [R004](../optimizations/R004-adaptive-draining.zh.md)
- [R019](../optimizations/R019-tail-estimation-and-sibling-ablation.zh.md)
- [R001](../optimizations/R001-runtime-request-rebalancing.zh.md)
- [R005](../optimizations/R005-queued-request-rebalancing.zh.md)
- [R012](../optimizations/R012-request-leases-and-migration.zh.md)
- [R013](../optimizations/R013-question-affinity-and-concurrency.zh.md)

这些条目是同一核心机制的组成部分，不代表各自独立创新。原 R 编号和历史证据保持不变。

## 局限、备选与负面结果

R019 同题长度乘数已删除；R013 亲和性未证明独立收益；R005 在已核对运行中未找到合格搬移；R001 新时机判断触发 0 次，默认关闭；R020 TP2
长尾对照是负面证据。

## 需要的独立验证

在相同共享、显存和权重发布机制下比较迁移执行开关；重复运行并报告 token 数、实际导入、事务成本、训练等待及质量。预测误差应与调度收益分开分析。

共享整体加速不等于本机制的独立收益或新颖性已确认。参见[当前认可策略与证据](../experiments/2026-10-19-r001-accepted-policy.zh.md)。

## 提议中的扩展

[R025：容量感知的动态推理并发控制](../optimizations/R025-adaptive-inference-concurrency.zh.md)
记录动态准入目标的研究提议，尚未实现或验证。

[R026：迁移到更大TP引擎加速长尾](../optimizations/R026-tail-acceleration-via-tp-migration.zh.md)
是新提议。首先必须确认预热后的更大TP引擎能缩短剩余decode时间，且收益足以抵消跨TP的KV重分片成本。R020仍保留为小模型负面对照。
