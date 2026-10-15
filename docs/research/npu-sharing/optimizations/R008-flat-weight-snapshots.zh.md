[English](R008-flat-weight-snapshots.md) | 中文

# R008: 扁平化、带版本的权重快照

状态：已实现；历史实验观察到发布成本下降。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

内部副本更新不必每次导出完整 HF 模型、配置和 tokenizer。先转推理 dtype，复制到独立拥有的扁平 CPU
映射文件，再原子发布带版本清单；加载前检查参数名、大小和边界，仍通过模型 load_weights 及昇腾后处理。正式 checkpoint 保留 HF 格式。

## 优化点与难点

减少序列化元数据和重复打包，同时保留不可变版本所有权。必须预分配文件空间避免稀疏 tmpfs
SIGBUS，训练权重卸载前快照必须真正拥有数据。仍有文件系统语义和约一份模型大小的 CPU 副本，不能称为普遍零复制或无磁盘。多 rank 汇集需要另行协调。

## 实测收益

16 次发布总耗时由 124.69 降到 83.16 秒，平均 7.79→5.20 秒，减少 33.3%；快照导出平均 2.01 秒。完整循环 459.42→426.58
秒，但仍慢于固定 397.32 秒。评估与采样回答不同，7.15% 总耗时下降不能视为严格组件消融。

## 下一步

对完全相同的不可变张量比较格式，再测端到端重叠及主机内存带宽；覆盖磁盘/tmpfs 容量不足和版本不匹配。

## 代码位置

- [areal/engine/weight_snapshot.py](../../../../areal/engine/weight_snapshot.py)

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/engine/vllm_ext/vllm_worker_extension.py](../../../../areal/engine/vllm_ext/vllm_worker_extension.py)

## 实验证据与会话来源

- [npu-share-fast01/comparison.json](../experiments/evidence/history/npu-share-fast01/comparison.json)

- [npu-share-ab01/comparison.json](../experiments/evidence/history/npu-share-ab01/comparison.json)

会话摘录编号：C016. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
