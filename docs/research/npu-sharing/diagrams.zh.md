# 研究机制图集

[English](diagrams.md) | [中文](diagrams.zh.md)

这些图是机制示意，不是实测时间线，方块宽度不表示耗时或内存大小。图中文字采用英文，便于后续论文使用；各中文文档配有中文解读。提案、负面实验及上游背景均明确标注。

[HTML gallery](diagrams.html)

## 总览

![R001–R008](optimizations/overview-1.png)

![R009–R016](optimizations/overview-2.png)

![R017–R024](optimizations/overview-3.png)

## 单项图解

### R001 · Rebalance running work only when the critical path wins

共享卡恢复后，先估算运行中请求的剩余时间，把暂停、重放、资源争用全部计入成本；只有关键路径净收益为正且约束满足才迁移。此项仍是提案。

[Note](optimizations/R001-runtime-request-rebalancing.zh.md) |
[PNG](optimizations/R001.png) | [SVG](optimizations/R001.svg)

### R002 · Reserve training memory first; admit inference into the remainder

先为训练状态、工作区和安全余量保留显存，再将剩余预算用于推理 KV。训练与推理保留独立权重，允许重叠不代表一定有请求可运行，也不保证计算不争用。

[Note](optimizations/R002-training-priority-concurrency.zh.md) |
[PNG](optimizations/R002.png) | [SVG](optimizations/R002.svg)

### R003 · Each inference replica resumes after its own version fence

固定推理卡完成自己的 HCCL 加载、确认版本后就恢复；共享卡独立等待快照加载完成再加入。图中两条发布路径逻辑独立，并非依次等待。

[Note](optimizations/R003-asymmetric-publication.zh.md) | [PNG](optimizations/R003.png)
| [SVG](optimizations/R003.svg)

### R004 · Drain or migrate: minimize the next training critical path

从最多三个已完成批次建立滑动窗口，分别预测整题回答收齐时间和训练卡就绪时间，取两者最大值作为下一次训练的启动时间，再比较继续排空与迁移。

[Note](optimizations/R004-adaptive-draining.zh.md) | [PNG](optimizations/R004.png) |
[SVG](optimizations/R004.svg)

### R005 · Rebalance queued requests with an acknowledged ownership handoff

优先处理公共等待请求，再检查从未执行的原生队列请求。原引擎确认取消之后才交给目标，确保同一时刻只有一个所有者；已运行请求不在此策略范围。

[Note](optimizations/R005-queued-request-rebalancing.zh.md) |
[PNG](optimizations/R005.png) | [SVG](optimizations/R005.svg)

### R006 · Reserve virtual space; map physical KV only as needed

保留逻辑地址和块编号，只给实际使用的 KV 映射物理页，并优先填充部分占用的 slab。空闲页面解除映射，不把虚拟地址空间等同于实际显存占用。

[Note](optimizations/R006-dynamic-kv-pages.zh.md) | [PNG](optimizations/R006.png) |
[SVG](optimizations/R006.svg)

### R007 · Borrow training capacity while rollout is waiting

等待训练批次时让训练卡参与推理，批次就绪时通过安全交接重新用于训练。训练进程仍持有优化器，训练分布式组没有被动态缩小或扩展。

[Note](optimizations/R007-single-npu-and-borrowing.zh.md) |
[PNG](optimizations/R007.png) | [SVG](optimizations/R007.svg)

### R008 · Publish a flat immutable weight snapshot instead of an HF package

内部权重更新只导出扁平化的推理精度权重，通过原子版本清单发布，省去重复打包配置和 tokenizer。快照必须独立拥有存储，不能被训练端后续改写。

[Note](optimizations/R008-flat-weight-snapshots.zh.md) | [PNG](optimizations/R008.png) |
[SVG](optimizations/R008.svg)

### R009 · Remove snapshot queue blocking from the HCCL producer

不是简单增加线程，而是拆开读取路径：HCCL 和快照线程独立读取已稳定的权重，不再由 HCCL 生产者向有界队列投递。下一次改写权重之前仍必须等待所有读取结束。

[Note](optimizations/R009-independent-snapshot-reader.zh.md) |
[PNG](optimizations/R009.png) | [SVG](optimizations/R009.svg)

### R010 · Let the main loop acquire work while the shared tail finishes

固定推理卡就绪后主循环立即获取下一批，快照、共享卡恢复及部分收尾在后台串行执行。到下一次修改训练权重、评估或退出之前仍需要汇合等待。

[Note](optimizations/R010-background-step-tail.zh.md) | [PNG](optimizations/R010.png) |
[SVG](optimizations/R010.svg)

### R011 · Eliminate final publication only when nobody needs it

只有确定训练真正结束且没有评估需要新权重时，才跳过无用的推理权重发布和恢复。最终模型保存、已有后台任务汇合及清理仍然执行。

[Note](optimizations/R011-terminal-publication-elision.zh.md) |
[PNG](optimizations/R011.png) | [SVG](optimizations/R011.svg)

### R012 · Continue a logical answer on another card by replaying its prefix

暂停后保留已生成 token、对数概率及版本信息，拿到卡2的租约后重放前缀、重新生成 KV 再继续。这里没有直接搬运 KV，也不能把跨版本续写等同于单一固定策略采样。

[Note](optimizations/R012-request-leases-and-migration.zh.md) |
[PNG](optimizations/R012.png) | [SVG](optimizations/R012.svg)

### R013 · Keep question groups together without confusing concurrency limits

同题回答初始放到同一个引擎，按未完成题组负载分配，但引擎容量仍按请求限制。图解是可选机制，不代表它已获得正收益；同题亲和性与已删除的同题长度预测是两回事。

[Note](optimizations/R013-question-affinity-and-concurrency.zh.md) |
[PNG](optimizations/R013.png) | [SVG](optimizations/R013.svg)

### R014 · Prewarm optimizer state; make memory queries observational

显存预算允许时提前异步创建优化器池和状态，优化器步数仍为零，不更新参数。内存状态查询去掉清缓存副作用，训练使用前等待预热完成。

[Note](optimizations/R014-optimizer-prewarm-and-pure-status.zh.md) |
[PNG](optimizations/R014.png) | [SVG](optimizations/R014.svg)

### R015 · Park inactive training cache and restore it before it is needed

仅解除完全空闲训练缓存段的物理映射，保留地址；这些无效数据无需复制到 RAM。利用等待回答的时间提前恢复映射，批次与内存都准备好才训练。

[Note](optimizations/R015-training-cache-parking-prefetch.zh.md) |
[PNG](optimizations/R015.png) | [SVG](optimizations/R015.svg)

### R016 · Warm actual Ascend execution paths before announcing readiness

在服务宣布就绪前预热真实模型和 block-table 路径，而不是无关的空算子。编译和加载成本移入启动阶段；比较时必须同时报告启动和稳态时间。

[Note](optimizations/R016-ascend-kernel-warmup.zh.md) | [PNG](optimizations/R016.png) |
[SVG](optimizations/R016.svg)

### R017 · Offload only enough complete allocations to cover the shortfall

先计算真实显存缺口，优先回收空闲工作区和 KV，再按完整分配块将必要的有效状态备份到 RAM。不是全部搬走，也不是运行到某层时才即时换层。

[Note](optimizations/R017-pressure-driven-partial-offload.zh.md) |
[PNG](optimizations/R017.png) | [SVG](optimizations/R017.svg)

### R018 · Retain old-version KV on the same card across training

在批次收齐后原卡暂停并保留旧版本 KV，显存不足才部分备份。训练后先恢复旧推理状态完成残留请求，再更新共享推理权重；避免新权重直接使用旧 KV。

[Note](optimizations/R018-retained-pause.zh.md) | [PNG](optimizations/R018.png) |
[SVG](optimizations/R018.svg)

### R019 · Estimate surviving tails without amplifying sibling bias

未完成回答属于截尾观测，按已生成长度估计条件剩余长度，并考虑活跃并发降低时 TPOT 的变化。去掉会共同放大同题回答预测偏差的乘数，同时保留失败证据。

[Note](optimizations/R019-tail-estimation-and-sibling-ablation.zh.md) |
[PNG](optimizations/R019.png) | [SVG](optimizations/R019.svg)

### R020 · Test TP=2 on identical long tails before building migration

使用相同长尾请求和固定剩余 token 数，在预热好的 TP1 与 TP2 实例分别重复测试。TP2 此处反而更慢，因此不应将双卡单实例迁移包装成已成功的优化。

[Note](optimizations/R020-tp-tail-negative-result.zh.md) | [PNG](optimizations/R020.png)
| [SVG](optimizations/R020.svg)

### R021 · Keep one sharing protocol; adapt backend memory and publication

共享协议复用，但运行时、内存池覆盖、状态预算和权重发布需要针对 Megatron 适配。先单批验证再对照，不把某一次后端实验结果当成普遍加速比。

[Note](optimizations/R021-megatron-backend-adaptation.zh.md) |
[PNG](optimizations/R021.png) | [SVG](optimizations/R021.svg)

### R022 · Per-rank memory decisions; all-rank readiness for TP execution

显存预算逐卡判断，宽裕的卡不搬，紧张的卡只处理自己的缺口；但 TP 实例必须等全部成员就绪才能执行。多卡不是多个独立训练任务，也不是弹性改变训练组成员。

[Note](optimizations/R022-multi-npu-rank-budgets.zh.md) | [PNG](optimizations/R022.png)
| [SVG](optimizations/R022.svg)

### R023 · Match conditions and trace causality before claiming a speedup

冻结代码、数据和参数，固定与共享保持一致的并发、日志及沙箱限制，记录请求、版本和设备事件，再生成四类图和双语总结。观测本身可能影响性能，固定组也不是未经修改的上游代码。

[Note](optimizations/R023-observability-and-reproducibility.zh.md) |
[PNG](optimizations/R023.png) | [SVG](optimizations/R023.svg)

### R024 · Only the consumed batch must finish before its optimizer update

训练只要求被消费的当前批次收齐，其他在途任务可以继续推理，在离策略限制内使用旧权重。异步 rollout 属于 AReaL 上游能力，不应写成此次新增创新。

[Note](optimizations/R024-async-and-multiversion-context.zh.md) |
[PNG](optimizations/R024.png) | [SVG](optimizations/R024.svg)

## 重新生成

```bash
python optimizations/generate_diagrams.py --font-dir /path/to/truetype/fonts
```

脚本从 `optimizations/diagram_specs.json` 生成 SVG；若已有 Pillow 和 TrueType 字体，还会输出 PNG 与总览图。SVG
无第三方依赖。
