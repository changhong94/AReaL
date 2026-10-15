[English](R007-single-npu-and-borrowing.md) | 中文

# R007: 单卡切换与借用空闲训练卡的基础能力

状态：已实现基础能力；少占一张卡不等于训练加速。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

固定分工中训练卡等 rollout 时闲置。npu_share
作为一个可选布尔开关统一角色切换与内存策略。单卡时交替训练推理；双卡时保留一张专用推理卡，训练卡在安全时加入推理池。优化器仍属于原训练进程，并非动态改变训练进程组规模。

## 难点与贡献边界

需要协调派发关闭、原生暂停确认、请求租约、显存准入、优化器连续性和版本提交。共享卡状态是一套协议，不只是设备放置开关。最初单卡实现并未接入两个参考运行时，slab
和映射随后才加入。AReaL 原有异步 rollout 属于上游能力，不能归为本工作新贡献。

## 证据与演进

单卡小实验在两次更新中保持优化器身份；压力实验连续完成四次 Adam 更新，覆盖常驻、仅 KV 回收和 CPU 卸载。16 次更新单卡共享 778.20 秒，双卡固定
370.05 秒：墙钟约 2.10 倍，预留卡秒约 1.05 倍。早期双卡共享还比固定慢 15.63%。后期组合迁移共享完成 100 批为 2380.91 秒，固定为
2715.66 秒，缩短 12.33%；策略与负载已经变化，不能归给最初的切换开关。

## 下一步

分别报告墙钟时间、初始化、预留设备秒、功耗和质量。长期稳定性与质量需要超过 smoke test 的验证；预留卡秒不等于实际计算占用。

## 代码位置

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/infra/borrowed_rollout.py](../../../../areal/infra/borrowed_rollout.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

- [areal/api/cli_args.py](../../../../areal/api/cli_args.py)

## 实验证据与会话来源

- [single-npu-timeshare-smoke03/validation-summary.json](../experiments/evidence/history/single-npu-timeshare-smoke03/validation-summary.json)

- [auto-memory-pressure01/validation-summary.json](../experiments/evidence/history/auto-memory-pressure01/validation-summary.json)

- [npu-share-comparison/comparison.json](../experiments/evidence/history/npu-share-comparison/comparison.json)

- [npu-share-ab01/comparison.json](../experiments/evidence/history/npu-share-ab01/comparison.json)

- `hundred-batches01`: 日志与脚本见持久历史归档。

会话摘录编号：C010, C013, C014, C015, C091.
[查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
