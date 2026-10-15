[English](R018-retained-pause.md) | 中文

# R018: 原卡暂停保留 KV 与部分备份

状态：已实现实验备选；按用户选择，迁移仍是最新默认方案。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

迁移浪费预填充，也可能让一个逻辑回答跨版本。完整批次就绪后用 keep 语义暂停，保留请求、旧权重和 KV。足够时有效页留在 HBM；压力下只备份所需有效页组到
RAM。训练后恢复旧版本完成保留回答，再加载共享卡新权重；专用卡可先发布。

## 难点与优化点

若还差一题就撤回，最后必需回答可能被暂停而导致死锁，因此本模式等待完整批次。保留块编号和地址，绝不能用旧 KV
配刚覆盖的新权重。长尾可能挡住共享发布和下次训练。它是重放成本与旧版本排空时间的交换，不是迁移的全面替代。

## 实测与局限

六批对照迁移 167.09 秒、原卡保留 164.82 秒，约少 1.36%，token 分别 135584/136631。保留模式有 52 次同卡同版本续写，没有卡 1 到卡
2 迁移。该训练显存足够，没有触发 RAM 备份；另一个原生测试才对 16 MiB 缓冲区搬移 4 MiB 并两轮逐字节恢复。不能把两项合并成部分卸载加速了该次训练的结论。

## 下一步

在真实压力和受控长尾下重复，同时统计省下的重放和推迟的发布。即使当前基线采用迁移，也保留此设计分支的记录。

## 代码位置

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

- [areal/engine/vllm_ext/dynamic_kv.py](../../../../areal/engine/vllm_ext/dynamic_kv.py)

- [areal/engine/vllm_ext/vllm_worker_extension.py](../../../../areal/engine/vllm_ext/vllm_worker_extension.py)

## 实验证据与会话来源

- [pause-retain-ab01/comparison.json](../experiments/evidence/history/pause-retain-ab01/comparison.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
