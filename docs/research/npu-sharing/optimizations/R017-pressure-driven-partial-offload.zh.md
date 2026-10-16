[English](R017-pressure-driven-partial-offload.md) | 中文

# R017: 压力驱动、按短缺量分块卸载常驻状态

状态：已实现安全交接边界的分块策略，不是任意算子执行时的缺页调度。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 机制示意图

![R017: Offload only enough complete allocations to cover the shortfall](R017.png)

**读图：** 先计算真实显存缺口，优先回收空闲工作区和 KV，再按完整分配块将必要的有效状态备份到 RAM。不是全部搬走，也不是运行到某层时才即时换层。

此图仅表达机制，不按时间或显存大小绘制；状态和结果边界以本文为准。

[SVG](R017.svg) | [PNG](R017.png)

## 问题与思路

整池卸载在下一阶段只缺少少量空间时也搬运大量状态。空间足够就保留常驻；不足时先回收空闲 workspace/KV，再把足够的完整优化器或权重分配块备份到
RAM；使用前只恢复缺失块并保持原地址。按块取整可能超过短缺字节，不是逐层流式搬运权重。

## 难点与优化点

短缺量依据实际空闲和配置上限，包含待增长的优化器、梯度和工作区。不能把分配器缓存全部当作保证可回收的自由空间。保持张量对象和优化器身份，旧主机副本不能覆盖新权重。有用数据需备份，空闲字节不需要；部分映射复制失败后禁止执行，不能
OOM 后重试半个 PPO。

## 证据与局限

早期整池压力实验用 null/48/37.5/37.5 GiB 有效预算且未改变物理显存，完成四次优化器更新，训练状态备份达到 18.75
GiB；检查对象地址和首尾采样字节，并非全张量校验。后续 pressure-memory02
保留了分块策略实验，但尚未恢复严格重复、同输入的分块对整池加速证据。功能验证不能包装成吞吐收益。

## 下一步

固定动态 KV 设置，扫描显存压力和分配粒度，报告搬运字节、切换延迟、RAM 峰值及总循环。正确标注 CaMem/ACL 基础能力来源，候选优化是策略集成。

## 代码位置

- [areal/engine/colocation_memory.py](../../../../areal/engine/colocation_memory.py)

- [areal/trainer/colocation_memory.py](../../../../areal/trainer/colocation_memory.py)

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/engine/vllm_ext/vllm_worker_extension.py](../../../../areal/engine/vllm_ext/vllm_worker_extension.py)

## 实验证据与会话来源

- [auto-memory-pressure01/validation-summary.json](../experiments/evidence/history/auto-memory-pressure01/validation-summary.json)

- `pressure-memory02`: 日志与脚本见持久历史归档。

会话摘录编号：C010, C080. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。

## 补充的本机分析产物

- [pressure-memory-timeline.json](../experiments/evidence/supplementary/pressure-memory-timeline.json)
