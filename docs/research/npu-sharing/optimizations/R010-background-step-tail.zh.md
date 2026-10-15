[English](R010-background-step-tail.md) | 中文

# R010: 共享发布与训练收尾移出主循环关键路径

状态：已实现；重叠已验证，首轮历史对照反而变慢。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

即使专用卡已恢复，同步快照、共享恢复、日志和清理仍可能挡住主循环进入
prepare_batch。把共享收尾排到串行执行器，专用卡就绪后主循环即可取下一批；下次修改训练参数、评估或退出前等待后台完成。对顺序敏感的数据清理提前到发布前。

## 难点与优化点

既要暴露并行机会，又不能让统计、快照和销毁发生竞争。后台异常必须唤醒取批次的等待者，而不是挂住。重叠阶段耗时不能相加当总墙钟；更早进入 prepare_batch
是机制生效，不代表整个训练更快。

## 实测结果

15 次普通批次交接全部与发布重叠。首次专用卡就绪 39.6149 秒，下一批开始取数 39.6160 秒，共享卡就绪 43.9598 秒。但最后发布/总循环为
377.51/408.47 秒，上一版为 356.99/389.89 秒，反而慢约 5.75%/4.76%。应保留这项负面结果，不能事后把所有后台化描述为加速。

## 下一步

区分暴露等待和有效重叠，用一致负载并检查评估与终止边界。本项为 R015 预取提供基础，但不能混用两者收益。

## 代码位置

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/trainer/training_prefetch.py](../../../../areal/trainer/training_prefetch.py)

## 实验证据与会话来源

- [npu-share-background02/comparison.json](../experiments/evidence/history/npu-share-background02/comparison.json)

会话摘录编号：C024, C025. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
