[English](README.md) | 中文

# NPU 共享研究记录

**图解入口：** [24 项机制示意图](diagrams.zh.md) · [可浏览图集](diagrams.html)。

本次从本会话原始消息、代码与历史实验恢复了 **24
个研究条目**。每项独立中英文文件，包含问题、机制、候选贡献、难点、代码位置、收益与局限、证据和下一步。全部条目及配图统一位于 optimizations；R001
仍明确标记为尚未实现的提案。

**不要把所有条目都视为已成功优化**：R001 尚未实现；R019 包含被移除策略；R020 是负面结果；R023 是实验基础设施；R024 是上游背景。原来的
R001–R006 编号保持不变。

先读：[证据方法与局限](methodology.zh.md) · [历史实验目录](experiments/history-catalog.zh.md) ·
[演进路线与论文候选贡献](roadmap.zh.md)

| 编号 | 研究条目                                                                                         | 当前结论                                                                  |
| ---- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| R001 | [成本感知的运行中请求再均衡](optimizations/R001-runtime-request-rebalancing.zh.md)               | 尚未实现；动机由 E001 支持，收益待验证。                                  |
| R002 | [训练显存优先的同卡训练与推理重叠](optimizations/R002-training-priority-concurrency.zh.md)       | 已实现，可选开启；实际重叠已验证，独立性能收益仍取决于负载。              |
| R003 | [专用推理卡与共享卡的非对称权重发布](optimizations/R003-asymmetric-publication.zh.md)            | 已实现；后续由 R009 和 R010 继续消除依赖。                                |
| R004 | [以训练就绪时间为目标的滑动窗口自适应排空](optimizations/R004-adaptive-draining.zh.md)           | 已实现；保留三批窗口，移除了同题长度缩放。                                |
| R005 | [尚未执行请求的成本感知再均衡](optimizations/R005-queued-request-rebalancing.zh.md)              | 已实现；已核对的两次对照中都没有实际搬移合格请求。                        |
| R006 | [slab 紧凑分配与 KV 物理页按需映射](optimizations/R006-dynamic-kv-pages.zh.md)                   | 已在限定昇腾布局实现；显存节省已验证，独立延迟收益尚未确立。              |
| R007 | [单卡切换与借用空闲训练卡的基础能力](optimizations/R007-single-npu-and-borrowing.zh.md)          | 已实现基础能力；少占一张卡不等于训练加速。                                |
| R008 | [扁平化、带版本的权重快照](optimizations/R008-flat-weight-snapshots.zh.md)                       | 已实现；历史实验观察到发布成本下降。                                      |
| R009 | [独立快照读取线程，解除 HCCL 队列阻塞](optimizations/R009-independent-snapshot-reader.zh.md)     | 已实现；短对照中发布阶段直接耗时改善。                                    |
| R010 | [共享发布与训练收尾移出主循环关键路径](optimizations/R010-background-step-tail.zh.md)            | 已实现；重叠已验证，首轮历史对照反而变慢。                                |
| R011 | [跳过最终批次无用途的发布与推理恢复](optimizations/R011-terminal-publication-elision.zh.md)      | 已实现，终止判断考虑评估需求。                                            |
| R012 | [基于租约的公共路由与跨卡续写](optimizations/R012-request-leases-and-migration.zh.md)            | 已实现，作为默认迁移路径；KV 重建而非直接传输。                           |
| R013 | [按题亲和调度与全局 rollout 并发](optimizations/R013-question-affinity-and-concurrency.zh.md)    | 题目亲和作为可选项已实现；一次实验中提高并发有效，亲和本身无额外收益。    |
| R014 | [优化器预热与无副作用的显存状态查询](optimizations/R014-optimizer-prewarm-and-pure-status.zh.md) | 已实现；同输入回放显示首批改善。                                          |
| R015 | [训练缓存保址解除映射与提前恢复](optimizations/R015-training-cache-parking-prefetch.zh.md)       | 实验开关已实现；机制已验证，计时回放没有证明严格数值等价。                |
| R016 | [推理服务就绪前预热真实昇腾请求内核](optimizations/R016-ascend-kernel-warmup.zh.md)              | 已针对受支持接口实现；固定和共享均生效。                                  |
| R017 | [压力驱动、按短缺量分块卸载常驻状态](optimizations/R017-pressure-driven-partial-offload.zh.md)   | 已实现安全交接边界的分块策略，不是任意算子执行时的缺页调度。              |
| R018 | [原卡暂停保留 KV 与部分备份](optimizations/R018-retained-pause.zh.md)                            | 已实现实验备选；按用户选择，迁移仍是最新默认方案。                        |
| R019 | [长尾预测与被移除的同题缩放策略](optimizations/R019-tail-estimation-and-sibling-ablation.zh.md)  | 保留尾部和动态服务修正；同题缩放已删除，启动前排序仍仅离线研究。          |
| R020 | [预热 TP=2 加速长尾的负面实验](optimizations/R020-tp-tail-negative-result.zh.md)                 | 已测纯推理备选；没有实现动态 TP 或 KV 迁移。                              |
| R021 | [共享执行的 Megatron 引擎适配](optimizations/R021-megatron-backend-adaptation.zh.md)             | 已实现并在记录的小模型配置验证。                                          |
| R022 | [多推理实例与 TP 逐卡显存预算](optimizations/R022-multi-npu-rank-budgets.zh.md)                  | 近期未提交扩展；4 卡 TP1、8 卡 TP2 有功能覆盖，尚无严格配对扩展收益结论。 |
| R023 | [可选观测与可复现对照测试工具](optimizations/R023-observability-and-reproducibility.zh.md)       | 已实现实验基础设施；跟踪本身可能影响性能。                                |
| R024 | [异步 rollout 与多版本实验：前置背景](optimizations/R024-async-and-multiversion-context.zh.md)   | 历史探索；必须区分上游能力与本项目扩展。                                  |

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

## 当前最新实验

[E001：Megatron c32 六批对照](experiments/20261015-144720-megatron-c32-b6.zh.md)
