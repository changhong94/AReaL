[English](R016-ascend-kernel-warmup.md) | 中文

# R016: 推理服务就绪前预热真实昇腾请求内核

状态：已针对受支持接口实现；固定和共享均生效。

整理日期：2026-10-15（America/Vancouver）。本页按历史证据整理，不代表重新实测。

## 问题与思路

请求图中一个请求运行很久、其他排队，主要是首次 Triton 编译与加载，不是 NPU 用数秒计算一个 token。持久化兼容的 Triton 缓存，并在服务就绪前预热
1、2、16 token 的真实模型和 block-table 路径。eager 不会关闭 JIT，已有二进制也需要进程内加载。

## 优化点与难点

把形状特化初始化移到正式请求前，同时单列初始化与训练循环耗时。预热必须走真实形状和接口，不是任意 dummy
算子；受支持路径失败则初始化失败，不支持的接口明确记跳过。缓存依赖编译器、设备与代码兼容性。这属于启动性能修复，不是通用预热算法创新。

## 实测收益

空缓存/复用缓存诊断首窗口约 10.63/0.154 秒，最长单 token 窗口 5.645/0.053 秒；5.645 秒中 RoPE 编译 3.776 秒、加载
1.826 秒，占 99.24%。加入预热后空缓存首窗口约 0.17 秒，最长单 token 0.049 秒，三组验证训练中无新增编译加载。空缓存预热约 17 秒、复用缓存约
0.125 秒。之后固定/共享循环 83.96/83.14 秒基本打平，且 token 不同。

## 下一步

分别报告冷暖缓存的启动及稳态成本；新模型、形状、后端重新验证覆盖。不要把主机 execute 窗口当作独占 NPU 计算时间。

## 代码位置

- [areal/engine/vllm_ext/ascend_kernel_warmup.py](../../../../areal/engine/vllm_ext/ascend_kernel_warmup.py)

- [areal/engine/vllm_ext/vllm_worker_extension.py](../../../../areal/engine/vllm_ext/vllm_worker_extension.py)

## 实验证据与会话来源

- [cold-start-diag02/comparison.json](../experiments/evidence/history/cold-start-diag02/comparison.json)

- [kernel-warmup02/comparison.json](../experiments/evidence/history/kernel-warmup02/comparison.json)

- [post-warmup-ab01/comparison.json](../experiments/evidence/history/post-warmup-ab01/comparison.json)

会话摘录编号：C069, C070, C071. [查看摘录](../experiments/evidence/conversation-excerpts.json)

原始证据根目录：`/root/areal-research-artifacts/history-20261015/`. 这是选择性归档，未包含模型检查点和完整 profiler
大文件；并非完整可运行备份。

## 论文表述约束

本页所说的“候选贡献”仅表示值得研究的机制，没有完成系统相关工作检索，不能据此声称首次提出。单次实时 RL
的回答长度、消费顺序和训练轨迹可能变化；不同实验的百分比不能相加。未单独消融的收益只能归于组合方案。
