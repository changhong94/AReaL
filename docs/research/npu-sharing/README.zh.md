[English](README.md) | 中文

# NPU 共享研究记录

**报告与论文：[report](report/README.md) · [PPT 大纲（待审核）](report/ppt-outline.zh.md)**

**主线入口：[三个核心机制](core-mechanisms.zh.md)**

1. **M1：训练优先的资源与显存共享**——安全借卡、按需显存、训练状态恢复。
1. **M2：面向训练就绪时间的请求调度**——剩余时间预测、排空、再均衡和 KV 迁移。
1. **M3：非对称异步权重发布**——逐副本就绪、独立发布、后台恢复。

R001–R025 保留为可追溯研究档案，不作为 25 个并列贡献。失败、无收益、未触发、备选方案和工程故障均保留，并明确测试条件和证据边界。工具 R023、上游背景 R024
单独定位；多后端与多卡是扩展支持。

R001 已实现；采用收紧策略的共享整体收益已验证（最近独立启动对照耗时降低 14.72%），KV
迁移的独立贡献仍待消融验证。参见[当前策略与证据](experiments/2026-10-19-r001-accepted-policy.zh.md)。

[机制图集](diagrams.zh.md) · [证据方法](methodology.zh.md) ·
[历史实验目录](experiments/history-catalog.zh.md) · [演进路线](roadmap.zh.md)

## 原始研究条目索引

| 编号 | 研究条目                                                                                           | 当前结论                                                                                                  |
| ---- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| R001 | [成本感知的运行中请求再均衡](optimizations/R001-runtime-request-rebalancing.zh.md)                 | 已实现；共享整体收益已验证，迁移独立贡献待消融。                                                          |
| R002 | [训练显存优先的同卡训练与推理重叠](optimizations/R002-training-priority-concurrency.zh.md)         | 已实现，可选开启；实际重叠已验证，独立性能收益仍取决于负载。                                              |
| R003 | [专用推理卡与共享卡的非对称权重发布](optimizations/R003-asymmetric-publication.zh.md)              | 已实现；后续由 R009 和 R010 继续消除依赖。                                                                |
| R004 | [以训练就绪时间为目标的滑动窗口自适应排空](optimizations/R004-adaptive-draining.zh.md)             | 已实现；保留三批窗口，移除了同题长度缩放。                                                                |
| R005 | [尚未执行请求的成本感知再均衡](optimizations/R005-queued-request-rebalancing.zh.md)                | 已实现；已核对的两次对照中都没有实际搬移合格请求。                                                        |
| R006 | [slab 紧凑分配与 KV 物理页按需映射](optimizations/R006-dynamic-kv-pages.zh.md)                     | 已在限定昇腾布局实现；显存节省已验证，独立延迟收益尚未确立。                                              |
| R007 | [单卡切换与借用空闲训练卡的基础能力](optimizations/R007-single-npu-and-borrowing.zh.md)            | 已实现基础能力；少占一张卡不等于训练加速。                                                                |
| R008 | [扁平化、带版本的权重快照](optimizations/R008-flat-weight-snapshots.zh.md)                         | 已实现；历史实验观察到发布成本下降。                                                                      |
| R009 | [独立快照读取线程，解除 HCCL 队列阻塞](optimizations/R009-independent-snapshot-reader.zh.md)       | 已实现；短对照中发布阶段直接耗时改善。                                                                    |
| R010 | [共享发布与训练收尾移出主循环关键路径](optimizations/R010-background-step-tail.zh.md)              | 已实现；重叠已验证，首轮历史对照反而变慢。                                                                |
| R011 | [跳过最终批次无用途的发布与推理恢复](optimizations/R011-terminal-publication-elision.zh.md)        | 已实现，终止判断考虑评估需求。                                                                            |
| R012 | [基于租约的公共路由与跨卡续写](optimizations/R012-request-leases-and-migration.zh.md)              | 已实现，作为默认迁移路径；KV 重建而非直接传输。                                                           |
| R013 | [按题亲和调度与全局 rollout 并发](optimizations/R013-question-affinity-and-concurrency.zh.md)      | 题目亲和作为可选项已实现；一次实验中提高并发有效，亲和本身无额外收益。                                    |
| R014 | [优化器预热与无副作用的显存状态查询](optimizations/R014-optimizer-prewarm-and-pure-status.zh.md)   | 已实现；同输入回放显示首批改善。                                                                          |
| R015 | [训练缓存保址解除映射与提前恢复](optimizations/R015-training-cache-parking-prefetch.zh.md)         | 实验开关已实现；机制已验证，计时回放没有证明严格数值等价。                                                |
| R016 | [推理服务就绪前预热真实昇腾请求内核](optimizations/R016-ascend-kernel-warmup.zh.md)                | 已针对受支持接口实现；固定和共享均生效。                                                                  |
| R017 | [压力驱动、按短缺量分块卸载常驻状态](optimizations/R017-pressure-driven-partial-offload.zh.md)     | 已实现安全交接边界的分块策略，不是任意算子执行时的缺页调度。                                              |
| R018 | [原卡暂停保留 KV 与部分备份](optimizations/R018-retained-pause.zh.md)                              | 已实现实验备选；按用户选择，迁移仍是最新默认方案。                                                        |
| R019 | [长尾预测与被移除的同题缩放策略](optimizations/R019-tail-estimation-and-sibling-ablation.zh.md)    | 保留尾部和动态服务修正；同题缩放已删除，启动前排序仍仅离线研究。                                          |
| R020 | [预热 TP=2 加速长尾的负面实验](optimizations/R020-tp-tail-negative-result.zh.md)                   | 已测纯推理备选；没有实现动态 TP 或 KV 迁移。                                                              |
| R021 | [共享执行的 Megatron 引擎适配](optimizations/R021-megatron-backend-adaptation.zh.md)               | 已实现并在记录的小模型配置验证。                                                                          |
| R022 | [多推理实例与 TP 逐卡显存预算](optimizations/R022-multi-npu-rank-budgets.zh.md)                    | 近期未提交扩展；4 卡 TP1、8 卡 TP2 有功能覆盖，尚无严格配对扩展收益结论。                                 |
| R023 | [可选观测与可复现对照测试工具](optimizations/R023-observability-and-reproducibility.zh.md)         | 已实现实验基础设施；跟踪本身可能影响性能。                                                                |
| R024 | [异步 rollout 与多版本实验：前置背景](optimizations/R024-async-and-multiversion-context.zh.md)     | 历史探索；必须区分上游能力与本项目扩展。                                                                  |
| R025 | [容量感知的动态推理并发控制](optimizations/R025-adaptive-inference-concurrency.zh.md)              | 容量策略已实现；复用引擎实验有正收益，正式确认待进行。                                                    |
| R026 | [迁移到更大TP引擎加速运行中长尾](optimizations/R026-tail-acceleration-via-tp-migration.zh.md)      | 研究提议；Qwen2-7B eager筛选为TP2更慢，跨TP KV迁移未实现。                                                |
| R027 | [基于滚动收益预测的推理切换](optimizations/R027-profit-aware-inference-return.zh.md)               | 可选原型已实现，开发验证中，尚未确认收益。                                                                |
| R028 | [多节点训练—推理资源共享与调度](optimizations/R028-multinode-training-inference-sharing.zh.md)     | 研究提议；跨节点扩展M1/M2/M3，尚未实现与验证。                                                            |
| R029 | [面向训练批次就绪的评分调度](optimizations/R029-training-ready-reward-scheduling.zh.md)            | 按题优先、跨题并行；槽位利用率超过99%、首批提前，整轮收益仍不稳定。                                       |
| R030 | [空闲推理卡加入训练与状态低开销重分片](optimizations/R030-idle-inference-training-expansion.zh.md) | 全量状态验证通过；最新切换15.31秒，开发比较RL快2.31%；token不同且不含初始化。保持可选，见多轮结果与限制。 |

## 维护规则

1. 改代码前分配稳定编号并写清假设。
1. 关联实现 commit 和实验编号。有未提交修改时，必须保留源码快照及哈希，不能只记录基础 commit。不要覆盖旧证据。
1. 保存配置、数据和模型身份、容器镜像、日志开关、资源分配、种子、失败和负面结果。声称训练输入相同时，须保留实际生成的回答。
1. 区分观测、解释、假设和已验证因果关系。
1. 做独立消融、多次重复并报告波动；同时记录训练质量和权重版本正确性。
1. 每次实验后更新记录。正确性和性能验证通过前，不把提议称为已实现的提升。
1. 声称论文创新前检索相关工作。原始证据保存在代码库外的持久存储，小体积摘要和清单放进代码库。
1. 每份文档维护英文原文件与 `.zh.md` 中文文件，互相链接并同步更新。代码注释继续用英文。

## 证据保存

E001 已复制到主机 `7.150.15.118` 的 `/root/areal-research-artifacts/`。
这是持久本地存储，不是独立的异机备份。完整路径和完整性清单见实验记录。 后续可使用
`--workspace-root /root/areal-research-artifacts/runs`；本次未更改脚本默认目录。

## 历史实验入口

[E001：Megatron c32 六批对照](experiments/20261015-144720-megatron-c32-b6.zh.md)
