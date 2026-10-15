[English](R024-async-and-multiversion-context.md) | 中文

# R024: 异步 rollout 与多版本实验：前置背景

状态：历史探索；必须区分上游能力与本项目扩展。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与范围

本会话先试多版本权重共存，再试下一批 rollout
与已收齐批次训练重叠。题目任务小于训练批次；更新只要求当前消费批次完整，其他在途任务可以在陈旧度约束内使用旧版本。这引出了后续借用空闲训练资源。

## 来源与设计边界

AReaL 已有异步 rollout 和 off-policy 控制，调整 max_head_offpolicyness
或展示批次并发不能视为原创。本工作后续扩展是逐副本就绪、显存共享和成本策略；逐 token 版本日志不代表已经实现永久版本绑定或直接 KV 传输。

## 证据与未决问题

已保留 multiversion/overlap/scale 目录支持演进顺序。本轮没有重建聊天中 419 对 147
秒草图的精确统计范围，因此不把它升级成严格主结果。早期两个副本仅共存数秒是可行性演示，不能证明额外占卡值得。

## 下一步

论文应明确上游异步基线，独立消融资源共享；核对逐 token 版本、重要性比率和陈旧度下的质量，不能把速度提升混成学习效果提升。

## 代码位置

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

## 实验证据与会话来源

- `multiversion-smoke01`: 日志与脚本见持久历史归档。

- `overlap-smoke01`: 日志与脚本见持久历史归档。

- `scale-smoke01`: 日志与脚本见持久历史归档。

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
