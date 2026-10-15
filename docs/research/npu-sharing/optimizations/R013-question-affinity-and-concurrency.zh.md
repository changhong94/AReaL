[English](R013-question-affinity-and-concurrency.md) | 中文

# R013: 按题亲和调度与全局 rollout 并发

状态：题目亲和作为可选项已实现；一次实验中提高并发有效，亲和本身无额外收益。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

训练消费完整题目组。逐回答分散可能留下很多未收齐题目，单实例并发太低也会浪费解码批处理。可选题目亲和按未完成题组负载初始放置同题所有回答，容量仍按请求计算；原卡关闭时才改归属，已完成回答不重跑。全局题目并发与每引擎
max_num_seqs 是不同限制。

## 难点与优化点

首选卡满时避免队首阻塞，并在奖励完成前维护题组归属，不能依赖诊断日志。增加在途工作也增加 KV 压力、陈旧度和可能无效的计算。题目亲和不同于 R019
删除的同题长度预测；删估计器不等于删题组语义。

## 实测收益与反例

共享三组：按请求 c16 为 85.22 秒，按请求 c32 为 70.81 秒，按题 c32 为 72.33 秒。48 道训练题初始同卡放置成功，但同为 c32 时按题慢
2.1%，主要收益来自并发。此前固定输出回放（128 请求，每个 256 token）还显示非单调扩展：总并发 64 时单/双副本 40.3/27.8 秒；并发 128 时反而
17.4/19.4 秒。副本和并发越多并非总越快。

## 下一步

对亲和、题目并发和引擎宽度做因子对照，先固定生成长度再跑实时 RL；保持题目集合、off-policy 上界和日志一致。

## 代码位置

- [areal/infra/request_router.py](../../../../areal/infra/request_router.py)

- [areal/infra/borrowed_rollout.py](../../../../areal/infra/borrowed_rollout.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

## 实验证据与会话来源

- `npu-share-taskaffinity01`: 日志与脚本见持久历史归档。

- `inference-fixed01`: 日志与脚本见持久历史归档。

会话摘录编号：C002, C034, C035. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。

## 补充的本机分析产物

- [inference-fixed-output-summary.json](../experiments/evidence/supplementary/inference-fixed-output-summary.json)
