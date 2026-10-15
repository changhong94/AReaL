[English](R005-queued-request-rebalancing.md) | 中文

# R005: 尚未执行请求的成本感知再均衡

状态：已实现；已核对的两次对照中都没有实际搬移合格请求。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

共享卡恢复时，剩余任务可能已归专用卡所有。恢复后独立评估一次，优先处理公共等待队列；公共队列为空时，才考虑专用实例中从未执行的原生排队请求。用冻结校准预测原地完成、目标完成及整体尾部，只有单请求和整体尾部收益均超过查询规划成本时才预留目标容量。

## 安全难点与创新边界

在 EngineCore 内原子取消，并重新检查 WAITING、零已计算/输出 token、从未调度标记；用内部尝试 ID 避免误取消重试。必须等源 HTTP
响应确认取消才释放租约，RPC 超时不能当作成功。目标版本和容量再次验证。这里只转移所有权，不传 KV；已经运行或曾被抢占的请求均不合格，因此与 R001 不同。

## 实测与缺口

对照为固定 175.36 秒、共享关闭再均衡 152.81 秒、开启 151.73 秒，但实际没有重新分配任何请求。因此 0.71% 的开关差异和 13.48%
的共享对固定差异都不能作为本功能收益。E001 再次出现五次 no_queued_work。这说明适用范围存在缺口，构成 R001 动机，但不证明 R001 必然有效。

## 下一步

先构造确有排队请求的负载，验证非零成功迁移、所有权和取消竞争；运行中迁移另做实验。保持日志开销相同，并报告每种跳过原因。

## 代码位置

- [areal/infra/queue_rebalance.py](../../../../areal/infra/queue_rebalance.py)

- [areal/infra/request_router.py](../../../../areal/infra/request_router.py)

- [areal/engine/vllm_ext/queue_rebalance.py](../../../../areal/engine/vllm_ext/queue_rebalance.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

## 实验证据与会话来源

- [queue-rebalance-comparison01/results-summary.json](../experiments/evidence/history/queue-rebalance-comparison01/results-summary.json)

会话摘录编号：C121. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
