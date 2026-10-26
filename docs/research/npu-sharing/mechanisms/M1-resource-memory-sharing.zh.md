[English](M1-resource-memory-sharing.md) | [总览](../core-mechanisms.zh.md)

# M1：训练优先的资源与显存共享

## 解决的问题与主要机制

什么时候借卡、显存不够怎么办，以及如何安全交还训练。

角色切换与训练优先准入 → 动态 KV 物理页 → 按缺口卸载 → 提前恢复训练状态。

## 可追溯的实现记录

- [R007](../optimizations/R007-single-npu-and-borrowing.zh.md)
- [R002](../optimizations/R002-training-priority-concurrency.zh.md)
- [R006](../optimizations/R006-dynamic-kv-pages.zh.md)
- [R017](../optimizations/R017-pressure-driven-partial-offload.zh.md)
- [R014](../optimizations/R014-optimizer-prewarm-and-pure-status.zh.md)
- [R015](../optimizations/R015-training-cache-parking-prefetch.zh.md)
- [R018](../optimizations/R018-retained-pause.zh.md)

这些条目是同一核心机制的组成部分，不代表各自独立创新。原 R 编号和历史证据保持不变。

## 局限、备选与负面结果

R018 原卡保留 KV 为非默认备选；R015 机制验证不等于严格数值等价或稳定加速。显存节省与端到端延迟收益应分别报告。

## 需要的独立验证

在相同请求与日志条件下分别关闭动态映射、部分卸载和预取；增加显存压力，记录物理显存、搬运量、切换时间、训练质量。

共享整体加速不等于本机制的独立收益或新颖性已确认。参见[当前认可策略与证据](../experiments/2026-10-19-r001-accepted-policy.zh.md)。
