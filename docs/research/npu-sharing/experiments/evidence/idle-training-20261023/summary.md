# Benchmark comparison

[English](summary.md) | [中文](summary.zh.md)

Run: `01`

## Configuration

- rl: code
- engine: fsdp
- concurrency: 16
- batches: 6
- batch_size: 32
- questions: 192
- answers: 8
- npu_count: 2
- inference_tp: 1
- inference_mode: graph
- max_head_offpolicyness: 1
- running_rebalance: kv
- adaptive_concurrency: capacity
- seed: 1
- created: 20261023-143705-0700

## Main comparison

| Metric                        |   Fixed |  Shared |
| ----------------------------- | ------: | ------: |
| Training loop (s)             | 460.921 | 866.057 |
| Questions / loop second       |   0.417 |   0.222 |
| Answers / loop second         |   3.332 |   1.774 |
| Reported generated tokens     |  364652 |  349163 |
| Reported tokens / loop second | 791.138 | 403.164 |

Shared loop time reduction: **-87.90%**. Speedup (fixed/shared): **0.532x**.

## Sampled device utilization

| Device metric (%) | Fixed mean | Shared mean |
| ----------------- | ---------: | ----------: |
| 1_aicore          |      59.36 |       29.16 |
| 1_hbm_used_pct    |      54.60 |       65.31 |
| 2_aicore          |      24.31 |       15.27 |
| 2_hbm_used_pct    |      48.88 |       32.19 |

## Measurement scope

- Timing covers the measured training loop, not environment preparation or chart
  rendering.
- Throughput divides counts by loop duration; it is not active-decode throughput.
- Generated answers and token counts can differ between arms. Speedup alone does not
  demonstrate equal model quality.
- Utilization is the arithmetic mean of recorded samples in the renderer's measurement
  window, not a time-weighted average. HBM includes reservations and caches.
- Token charts reconstruct decode-window events; audit discrepancies are retained below.

## Token audit

- Fixed:
  `{"reconstructed": 364633, "reported": 364652, "segment_differences": {"generate-tokens-011a3f671f4d4eafb4c5ef1df541c89c-a57e2de9": -1, "generate-tokens-0af39a584a2b4260b28fbe7ebc307986-9c3a95e9": -1, "generate-tokens-1bcfe32240d94fc6b25f5cd6489f91cf-b26471e3": -1, "generate-tokens-27b5823ed01c414197056bb971ebd4b0-8fe934c4": -1, "generate-tokens-3306b897698a4c359334559688f96368-b444536e": -1, "generate-tokens-361dc2d305c3454d888c4b3961e974cf-870e466d": -1, "generate-tokens-38044eccf2d5428da85ce84ec0729a8e-aa12aa07": -1, "generate-tokens-3df9adb58a72422e850c79886a85c92d-bcc8001a": -1, "generate-tokens-3f18e1de0692418fa84c6b252f03387c-9edd1ccb": -1, "generate-tokens-401e4a5e21a44e2e92139e54ea8914df-b1404e6f": -1, "generate-tokens-48f20f94a4784a4d9f12d7449983d414-8a84f015": -1, "generate-tokens-6456edf4173e4c3fb1cec284fa348dbd-82a451af": -1, "generate-tokens-6ba537112d5a45d0a168bf53ad8d0751-b9f6e38a": -1, "generate-tokens-7c04b96498c74ff39440abfcbb32c4b2-89ef3c2f": -1, "generate-tokens-a616170cfc4b4f7f9a9dee09c40c10dd-83801d6b": -1, "generate-tokens-c5fa77ad50f14ccf85acd3a2c9beeab6-b33fb3ab": -1, "generate-tokens-c792c7d1017f4c8593c68d83438a8646-ad4a47d3": -1, "generate-tokens-eb15846e86db47ba949093eba5b53aff-89b28a60": -1, "generate-tokens-eb44133acb644bc3a501a411aa868caf-9028fa9b": -1}}`
- Shared:
  `{"reconstructed": 349153, "reported": 349163, "segment_differences": {"generate-tokens-23a042886275461295fcc90e091d4567-a46c6222": -1, "generate-tokens-2e5c7900e3ec49a1a66b782d54fb04cd-99d7e2ac": -1, "generate-tokens-6431346bfc4b42bd8b3fb36cf76f4071-9cc6f102": -1, "generate-tokens-773b9e1ba63c4d79bda1680034b65774-864d1624": -1, "generate-tokens-7c13889ce03e4aff8f9aebc7c45d1a83-ac7667d9": -1, "generate-tokens-7cb937c618ce416f8f599378a112ba40-87356915": -1, "generate-tokens-882dd174f4094947b2e90d7a04697264-98d30b69": -1, "generate-tokens-98a4cd4010f647e4b206abd4a50c2fed-95e76f3c": -1, "generate-tokens-a68e501f836345a2a61239f09d995524-b7e4ce89": -1, "generate-tokens-b07991c4e8aa4ecda82548f155bf9a03-b4b0d3c2": -1, "generate-tokens-b5b243ad2c7b4f8a8b330e9eccbe218d-82f491a5": 1, "generate-tokens-b98900d5f81f4a3eaaaf4e774b0832df-a8083071": -1, "generate-tokens-d25c57c57071435fa6cd86003e3d60aa-9a8027d5": -1, "generate-tokens-f2db1c75567243d39dcd89f72ea636bb-93e8b1d2": 1}}`

[Charts](index.html) | [Machine-readable results](summary.json)
