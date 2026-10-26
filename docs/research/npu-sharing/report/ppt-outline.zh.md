# PPT 大纲：异步 RL 的 NPU 共享

**v0.2 审阅稿，2026-10-21。用户已要求根据最新情况修订并生成 PPT；PPT 等待审核，尚未定稿。**

英文 PPT，18 页正文（含封面）和 2 页附录，面向内部系统与 RL 研究人员，约 20–25 分钟。

[English](ppt-outline.md)

## 相比 v0.1 的修改

- 保留 M1/M2/M3 三个核心机制，R001–R028 作为可追溯记录。
- 增加 R025 当前并发公式、R027 收益判断和训练缓存保留。
- 更新完整推理图模式、等 TP、多实例和暂停 KV 的验证边界。
- 分别呈现历史 14.72%、R025 开发结果 15.40%、最新正式结果 9.01%/8.96%，不混合归因。
- 记录 Megatron/R027 修复：30 项 CPU 测试通过，NPU 复测待完成。
- 保留负面证据，明确 R026/R028 尚属提案。

## 01. 异步 RL 的 NPU 共享

围绕资源共享、请求调度和权重发布三个核心机制，汇报最新实现与证据。

来源：[core-mechanisms.md](../core-mechanisms.md)

## 02. 目标是让训练更早推进

目标是缩短训练循环，不能仅以忙碌率或卡1产生的 token 数判断收益。

来源：[methodology.md](../methodology.md),
[R007-single-npu-and-borrowing.md](../optimizations/R007-single-npu-and-borrowing.md)

## 03. 三个核心机制

M1 提供安全可用的资源，M2 调度有效工作，M3 控制版本就绪。R025 和 R027 是这三个机制的扩展。

来源：[core-mechanisms.md](../core-mechanisms.md),
[R025-adaptive-inference-concurrency.md](../optimizations/R025-adaptive-inference-concurrency.md),
[R027-profit-aware-inference-return.md](../optimizations/R027-profit-aware-inference-return.md)

## 04. M1：安全借用训练卡

改变设备角色，不动态缩减训练进程组。

来源：[R007-single-npu-and-borrowing.md](../optimizations/R007-single-npu-and-borrowing.md)

## 05. M1：按需求回收显存

动态 KV 映射、部分卸载和预取尽量保留有效状态，同时保留内存预算检查。

来源：[R017-pressure-driven-partial-offload.md](../optimizations/R017-pressure-driven-partial-offload.md),
[R006-dynamic-kv-pages.md](../optimizations/R006-dynamic-kv-pages.md),
[R015-training-cache-parking-prefetch.md](../optimizations/R015-training-cache-parking-prefetch.md)

## 06. M2：分别预测数据和设备就绪

使用最多三个已完成 batch 的统计窗口。训练开始需要题目答案收齐，也需要训练设备就绪。

来源：[R004-adaptive-draining.md](../optimizations/R004-adaptive-draining.md),
[R019-tail-estimation-and-sibling-ablation.md](../optimizations/R019-tail-estimation-and-sibling-ablation.md)

## 07. M2：有限探索的运行中 KV 迁移

R001 优先处理排队请求，再考虑运行中请求。探索策略为每次最多5个请求、两次尝试、累计事务耗时达到2秒后停止新探索；正收益计划单独处理。

来源：[R001-runtime-request-rebalancing.md](../optimizations/R001-runtime-request-rebalancing.md),
[2026-10-19-r001-accepted-policy.md](../experiments/2026-10-19-r001-accepted-policy.md)

## 08. R025：并发数随可用能力变化

当前公式 ceil(C×R/F)。C 为基准题目并发，R 为就绪实例数，F 为固定实例数；TP rank 不是独立实例。

来源：[R025-adaptive-inference-concurrency.md](../optimizations/R025-adaptive-inference-concurrency.md)

## 09. R027：有收益才切回推理

训练结束时，根据下一批就绪情况和近期切换代价判断是否回推理。样本不足时正常回推理以校准；正式结果主要验证下一批已就绪的快捷路径。

来源：[R027-profit-aware-inference-return.md](../optimizations/R027-profit-aware-inference-return.md),
[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md)

## 10. M3：各实例独立完成权重发布

HCCL 与快照独立读取，各分支确认完成后独立放行；下一次修改权重前等待相关读者完成。

来源：[R009-independent-snapshot-reader.md](../optimizations/R009-independent-snapshot-reader.md),
[R010-background-step-tail.md](../optimizations/R010-background-step-tail.md)

## 11. 正确性约束定义迁移边界

唯一所有权、版本一致性和每卡预算优先。同版本暂停请求的 TP1/TP2 探针均保持256个 token 和对应概率一致。

来源：[R027-profit-aware-inference-return.md](../optimizations/R027-profit-aware-inference-return.md),
[R001-runtime-request-rebalancing.md](../optimizations/R001-runtime-request-rebalancing.md)

## 12. 开发比较与正式确认分开

复用引擎加快开发，正式确认独立启动；冻结源码、工作量和观测开关。

来源：[methodology.md](../methodology.md),
[R023-observability-and-reproducibility.md](../optimizations/R023-observability-and-reproducibility.md)

## 13. 早期共享系统结果：循环时间减少14.72%

2026-10-19 独立启动：179.232秒与152.847秒，各576回答，137098/138424 tokens，10次确认导入。历史配置，不能直接等同于当前图模式或
R001 单独收益。

来源：[2026-10-19-r001-accepted-policy.md](../experiments/2026-10-19-r001-accepted-policy.md),
[accepted.json](../experiments/evidence/r001-moderated-20261019/accepted.json)

## 14. R025 开发结果：提高有效并发

两次复用引擎比较，静态16平均181.08秒，动态153.20秒，减少15.40%，tokens减少1.72%。仍需静态32和独立启动控制。

来源：[2026-10-20-r025-capacity-admission.md](../experiments/2026-10-20-r025-capacity-admission.md)

## 15. 图模式正式确认：循环时间减少约9%

两轮独立启动，FSDP、推理完整图模式、训练eager，96题、每题6回答、每批16题、并发16、6批、offpolicy2。节省9.01%和8.96%，合计8.99%。

来源：[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md),
[2026-10-21-r027-formal.json](../experiments/2026-10-21-r027-formal.json),
[2026-10-21-r027-formal-repeat.json](../experiments/2026-10-21-r027-formal-repeat.json)

## 16. 实现支持与验证范围不同

图模式已在记录的双卡FSDP工作量下验证；多实例等TP探针不证明扩展收益。今日Megatron R027修复通过30项CPU测试，等待NPU复测。

来源：[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md),
[2026-10-19-r001-tp2-megatron.md](../experiments/2026-10-19-r001-tp2-megatron.md),
[R001-runtime-request-rebalancing.md](../optimizations/R001-runtime-request-rebalancing.md)

## 17. 负面结果指导后续设计

保留同题缩放失效、门控未触发、内存管理开销及TP结论依赖场景的记录。7B图模式TP2测试改善15.21–28.57%，跨TP在线迁移尚未实现。

来源：[R019-tail-estimation-and-sibling-ablation.md](../optimizations/R019-tail-estimation-and-sibling-ablation.md),
[R020-tp-tail-negative-result.md](../optimizations/R020-tp-tail-negative-result.md),
[2026-10-20-r026-qwen2-7b-graph-tp-screen.md](../experiments/2026-10-20-r026-qwen2-7b-graph-tp-screen.md),
[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md)

## 18. 面向论文的下一步证据

当前最可靠的结论是特定工作量下整体系统收益；下一步做三个机制消融、更强基线、质量验证和扩展。

来源：[roadmap.md](../roadmap.md), [methodology.md](../methodology.md),
[R026-tail-acceleration-via-tp-migration.md](../optimizations/R026-tail-acceleration-via-tp-migration.md),
[R028-multinode-training-inference-sharing.md](../optimizations/R028-multinode-training-inference-sharing.md)

## 19. 附录：当前 benchmark 默认值

当前默认32题×8回答不同于正式证据的16×6；coverage覆盖多种大小。

来源：[README.md](../../../../benchmark/npu_share/README.md),
[COVERAGE_PLAN.md](../../../../benchmark/npu_share/COVERAGE_PLAN.md)

## 20. 附录：证据和实现记录

证据来源为研究目录；保留历史事实，同时明确最新状态。

来源：[README.md](../report/README.md)

## 待审核

- [ ] 叙事顺序和技术深度。
- [ ] 三个核心机制的表达。
- [ ] 结果边界、负面证据和后续研究。
- [ ] 图表、文字和版式。
- [ ] 用户确认后再标记定稿。
