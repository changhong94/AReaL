[English](R012-request-leases-and-migration.md) | 中文

# R012: 基于租约的公共路由与跨卡续写

状态：已实现，作为默认迁移路径；KV 重建而非直接传输。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 机制示意图

![R012: Continue a logical answer on another card by replaying its prefix](R012.png)

**读图：** 暂停后保留已生成 token、对数概率及版本信息，拿到卡2的租约后重放前缀、重新生成 KV 再继续。这里没有直接搬运
KV，也不能把跨版本续写等同于单一固定策略采样。

此图仅表达机制，不按时间或显存大小绘制；状态和结果边界以本文为准。

[SVG](R012.svg) | [PNG](R012.png)

## 问题与思路

请求固定在共享卡会等待整段训练。改为每段生成从公共队列取得引擎租约。撤回时关闭准入、暂停原生推理并等待明确返回；保留输出 token、log probability 和逐
token 版本，再取得另一引擎并重放累计前缀。CPU 工作流与奖励的归属不必等于实际生成卡。

## 正确性与难点

原生入队与暂停需要串行化，迟到请求收到明确重试语义。HTTP 超时可能导致执行状态不明，不能直接另起一份造成重复。续写可跨权重版本并保留逐 token
来源，但不等价于同一策略完整采样；要保持重要性权重和 off-policy 语义并限制陈旧度。当前没有直接跨设备 KV 传输。

## 实测与局限

历史跟踪证明卡 1 到卡 2 续写成功且逻辑回答完整。后期迁移组完成 576 个回答，其中 152 个跨卡，总耗时 167.09 秒；100 批组合策略总耗时少
12.33%。两者都不能隔离路由自身收益。前缀重放浪费计算，分卡后批量过小也可能抵消增加设备的收益，见 R013。

## 下一步

把重复前缀 token、所有权交接空档与最终输出分别计量，补异常注入和质量对比；R018 比较同版本原卡保留，R001 探索反向再均衡。

## 代码位置

- [areal/infra/request_router.py](../../../../areal/infra/request_router.py)

- [areal/infra/borrowed_rollout.py](../../../../areal/infra/borrowed_rollout.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

- [areal/engine/vllm_ext/areal_vllm_server.py](../../../../areal/engine/vllm_ext/areal_vllm_server.py)

## 实验证据与会话来源

- `npu-share-taskaffinity01`: 日志与脚本见持久历史归档。

- [pause-retain-ab01/comparison.json](../experiments/evidence/history/pause-retain-ab01/comparison.json)

- `hundred-batches01`: 日志与脚本见持久历史归档。

会话摘录编号：C030, C034, C091. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
