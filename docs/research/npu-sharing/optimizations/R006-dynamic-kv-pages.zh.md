[English](R006-dynamic-kv-pages.md) | 中文

# R006: slab 紧凑分配与 KV 物理页按需映射

状态：已在限定昇腾布局实现；显存节省已验证，独立延迟收益尚未确立。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 机制示意图

![R006: Reserve virtual space; map physical KV only as needed](R006.png)

**读图：** 保留逻辑地址和块编号，只给实际使用的 KV 映射物理页，并优先填充部分占用的 slab。空闲页面解除映射，不把虚拟地址空间等同于实际显存占用。

此图仅表达机制，不按时间或显存大小绘制；状态和结果边界以本文为准。

[SVG](R006.svg) | [PNG](R006.png)

## 问题与思路

逻辑块动态分配不等于物理显存按需分配。历史实例预留约 21.2 GiB KV，逻辑使用率采样却只有 0.9%–2.1%。先采用 FineServe
式紧凑布局，优先填已部分使用的 slab，再保留虚拟地址范围，只为实际保留的块编号映射 ACL 物理页。同步后回收确认空闲的整组物理页，保留少量水位减少抖动。

## 来源、优化点与难点

FineServe 提供 slab 分配思想及相应代码，保留 BSD 归属；distributed-runtime/CaMem
提供保址物理映射参考。这里的扩展是结合昇腾原生逐层 K/V 布局与 RL 交接，而非发明 slab 或虚拟内存。所检查的 FineServe
版本本身仍预分配大张量。必须保护全部保留请求而非仅当前调度集合，固定 null block、清零新页，部分失败后禁止恢复；物理页粒度不等于逻辑 slab。

## 实测收益与边界

动态 KV 验证中每实例初始物理 KV 为 112 MiB，两实例峰值为 336/784 MiB，历史整池约 21.2 GiB。192 个回答全部完成，循环 84.30
秒。固定与动态共享对照为 82.79/83.97 秒、45604/47824 token；卡 2 平均 HBM 从 44.2% 降到
12.6%。该对比同时改变共享与分配方式，不能隔离分配器速度或开销。省显存可支持更大负载，但不会自动加速小负载。

## 下一步

在共享内部开关 AREAL_DYNAMIC_KV，保持工作量一致，分别计时映射、清零、同步、解除映射，并增大上下文与并发到节省显存真正有价值。TP 扩展依赖 R022 的逐
rank 边界，不能由此推断任意模型、KV 复制路径和图捕获都受支持。

## 代码位置

- [areal/engine/vllm_ext/kv_slab.py](../../../../areal/engine/vllm_ext/kv_slab.py)

- [areal/engine/vllm_ext/kv_slab_adapter.py](../../../../areal/engine/vllm_ext/kv_slab_adapter.py)

- [areal/engine/vllm_ext/dynamic_kv.py](../../../../areal/engine/vllm_ext/dynamic_kv.py)

- [areal/engine/vllm_ext/dynamic_kv_allocator.cpp](../../../../areal/engine/vllm_ext/dynamic_kv_allocator.cpp)

- [licenses/FineServe.txt](../../../../licenses/FineServe.txt)

## 实验证据与会话来源

- [kv-slab-smoke03/validation-summary.json](../experiments/evidence/history/kv-slab-smoke03/validation-summary.json)

- [dynamic-kv03/summary.json](../experiments/evidence/history/dynamic-kv03/summary.json)

- [dynamic-kv-ab01/comparison.json](../experiments/evidence/history/dynamic-kv-ab01/comparison.json)

会话摘录编号：C008, C079, C080, C081, C082.
[查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
