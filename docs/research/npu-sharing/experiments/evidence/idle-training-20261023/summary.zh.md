# 基准测试对比

[English](summary.md) | [中文](summary.zh.md)

实验目录: `01`

## 实验配置

- RL 任务 (`rl`): code
- 训练引擎 (`engine`): fsdp
- 题目并发数 (`concurrency`): 16
- 批次数 (`batches`): 6
- 每批题数 (`batch_size`): 32
- 题目数 (`questions`): 192
- 每题回答数 (`answers`): 8
- NPU 数量 (`npu_count`): 2
- 推理 TP (`inference_tp`): 1
- 推理执行模式 (`inference_mode`): graph
- 最大头部离策略限制 (`max_head_offpolicyness`): 1
- 共享组运行中请求再均衡 (`running_rebalance`): kv
- 共享组动态并发策略 (`adaptive_concurrency`): capacity
- 随机种子 (`seed`): 1
- 创建时间 (`created`): 20261023-143705-0700

## 主要指标比较

| 指标                      |  固定组 |  共享组 |
| ------------------------- | ------: | ------: |
| 训练循环耗时（秒）        | 460.921 | 866.057 |
| 题目吞吐量（题/秒）       |   0.417 |   0.222 |
| 回答吞吐量（条/秒）       |   3.332 |   1.774 |
| 日志记录的生成 token 总数 |  364652 |  349163 |
| 生成吞吐量（token/秒）    | 791.138 | 403.164 |

共享组训练循环耗时降低: **-87.90%**. 加速比（固定组耗时/共享组耗时）: **0.532x**.

## 设备利用率采样统计

| 设备指标（%）  | 固定组均值 | 共享组均值 |
| -------------- | ---------: | ---------: |
| 1_aicore       |      59.36 |      29.16 |
| 1_hbm_used_pct |      54.60 |      65.31 |
| 2_aicore       |      24.31 |      15.27 |
| 2_hbm_used_pct |      48.88 |      32.19 |

## 统计口径

- 耗时覆盖实测训练循环，不包括环境准备和图表生成。
- 吞吐量使用数量除以训练循环耗时，不是仅在解码执行期间的吞吐量。
- 两组生成的回答及 token 数可能不同；加速比本身不能证明模型质量相同。
- 利用率是图表统计窗口内采样值的算术平均，不是时间加权平均；显存占用包括预留空间和缓存。
- Token 图根据解码窗口事件重建，核对差异保留在下方。

## Token 核对

- 固定组:
  `{"reconstructed": 364633, "reported": 364652, "segment_differences": {"generate-tokens-011a3f671f4d4eafb4c5ef1df541c89c-a57e2de9": -1, "generate-tokens-0af39a584a2b4260b28fbe7ebc307986-9c3a95e9": -1, "generate-tokens-1bcfe32240d94fc6b25f5cd6489f91cf-b26471e3": -1, "generate-tokens-27b5823ed01c414197056bb971ebd4b0-8fe934c4": -1, "generate-tokens-3306b897698a4c359334559688f96368-b444536e": -1, "generate-tokens-361dc2d305c3454d888c4b3961e974cf-870e466d": -1, "generate-tokens-38044eccf2d5428da85ce84ec0729a8e-aa12aa07": -1, "generate-tokens-3df9adb58a72422e850c79886a85c92d-bcc8001a": -1, "generate-tokens-3f18e1de0692418fa84c6b252f03387c-9edd1ccb": -1, "generate-tokens-401e4a5e21a44e2e92139e54ea8914df-b1404e6f": -1, "generate-tokens-48f20f94a4784a4d9f12d7449983d414-8a84f015": -1, "generate-tokens-6456edf4173e4c3fb1cec284fa348dbd-82a451af": -1, "generate-tokens-6ba537112d5a45d0a168bf53ad8d0751-b9f6e38a": -1, "generate-tokens-7c04b96498c74ff39440abfcbb32c4b2-89ef3c2f": -1, "generate-tokens-a616170cfc4b4f7f9a9dee09c40c10dd-83801d6b": -1, "generate-tokens-c5fa77ad50f14ccf85acd3a2c9beeab6-b33fb3ab": -1, "generate-tokens-c792c7d1017f4c8593c68d83438a8646-ad4a47d3": -1, "generate-tokens-eb15846e86db47ba949093eba5b53aff-89b28a60": -1, "generate-tokens-eb44133acb644bc3a501a411aa868caf-9028fa9b": -1}}`
- 共享组:
  `{"reconstructed": 349153, "reported": 349163, "segment_differences": {"generate-tokens-23a042886275461295fcc90e091d4567-a46c6222": -1, "generate-tokens-2e5c7900e3ec49a1a66b782d54fb04cd-99d7e2ac": -1, "generate-tokens-6431346bfc4b42bd8b3fb36cf76f4071-9cc6f102": -1, "generate-tokens-773b9e1ba63c4d79bda1680034b65774-864d1624": -1, "generate-tokens-7c13889ce03e4aff8f9aebc7c45d1a83-ac7667d9": -1, "generate-tokens-7cb937c618ce416f8f599378a112ba40-87356915": -1, "generate-tokens-882dd174f4094947b2e90d7a04697264-98d30b69": -1, "generate-tokens-98a4cd4010f647e4b206abd4a50c2fed-95e76f3c": -1, "generate-tokens-a68e501f836345a2a61239f09d995524-b7e4ce89": -1, "generate-tokens-b07991c4e8aa4ecda82548f155bf9a03-b4b0d3c2": -1, "generate-tokens-b5b243ad2c7b4f8a8b330e9eccbe218d-82f491a5": 1, "generate-tokens-b98900d5f81f4a3eaaaf4e774b0832df-a8083071": -1, "generate-tokens-d25c57c57071435fa6cd86003e3d60aa-9a8027d5": -1, "generate-tokens-f2db1c75567243d39dcd89f72ea636bb-93e8b1d2": 1}}`

[图表](index.html) | [机器可读结果](summary.json)
