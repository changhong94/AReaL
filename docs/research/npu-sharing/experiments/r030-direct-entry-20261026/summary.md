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
- created: 20261026-000110-0700

## Main comparison

| Metric                        |   Fixed |  Shared |
| ----------------------------- | ------: | ------: |
| Training loop (s)             | 463.192 | 466.736 |
| Questions / loop second       |   0.415 |   0.411 |
| Answers / loop second         |   3.316 |   3.291 |
| Reported generated tokens     |  360875 |  359324 |
| Reported tokens / loop second | 779.104 | 769.866 |

Shared loop time reduction: **-0.76%**. Speedup (fixed/shared): **0.992x**.

## Sampled device utilization

| Device metric (%) | Fixed mean | Shared mean |
| ----------------- | ---------: | ----------: |
| 1_aicore          |      58.91 |       57.63 |
| 1_hbm_used_pct    |      54.97 |       67.89 |
| 2_aicore          |      23.41 |       29.60 |
| 2_hbm_used_pct    |      48.83 |       30.73 |

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
  `{"reconstructed": 360861, "reported": 360875, "segment_differences": {"generate-tokens-1dda25b9e8624b8f9c805d34171f3818-bd6ea305": -1, "generate-tokens-259fe86a097c47dc892f139d001d721f-a526ef9f": 1, "generate-tokens-387c3ae454bb4f409ac35609cde2e3fb-991e673c": -1, "generate-tokens-3f5bcef60e5c4cd399b8787096cc68d5-9d4b2b31": -1, "generate-tokens-472d4956ae034cdf8fef20345faa8694-b07e5ee1": -1, "generate-tokens-495c5f205f9544019a6a56f46fabb6dc-ad7bb76e": -1, "generate-tokens-4a9db7e45f904734a31490d4bcc64b22-a1fa865b": -1, "generate-tokens-5ce82e3fbd934d6b932565deb68e490e-98050534": -1, "generate-tokens-5fa9df75f4784aa0a32716122bed7192-a70196b8": -1, "generate-tokens-5fefa532508049d7962a91c7115d13ca-adaf70e0": -1, "generate-tokens-75eb4dec263440bf90dfbb5f135eaaba-99b01458": -1, "generate-tokens-9596f9f7ba68431b93f7d4b5b5752c57-bd2188d2": -1, "generate-tokens-95bfafd4d8ac4194a69a4d6fcdb247a2-b3c76d85": -1, "generate-tokens-a460db860d60475f961e96a23e4dfc6f-afaa3b77": -1, "generate-tokens-bafc7813356347f78ce65b1b367e741d-b4cf70d1": -1, "generate-tokens-d92cbc862d0b4b149c73c6011446bcfe-9f629436": -1}}`
- Shared:
  `{"reconstructed": 359313, "reported": 359324, "segment_differences": {"generate-tokens-1cf7e3b09e8e43dc9f9d89a846c6562d-af51cc74": -1, "generate-tokens-20a39f9569114a588e6c773e22d3e64c-b7959638": -1, "generate-tokens-69b91ceda7ad440b939f0bf38fb23971-b78c7cf8": -1, "generate-tokens-7195cb93751f4c3eb44704d43c9e632d-aa334324": -1, "generate-tokens-75f72bd57592485eacd9be79b2e3b87e-9e7cb66e": -1, "generate-tokens-919c1a658fd344b2a61f2e5551cc2d8e-9eebce79": -1, "generate-tokens-9846546592754ec385fda9a607995c10-8b686d25": 1, "generate-tokens-a0bd522e00e14cf7a917a3c4fec4550d-bcb3fc97": -1, "generate-tokens-adc950bc38434f72bb1e7e7b1f7ae5fc-b920b53b": -1, "generate-tokens-b74bdaec51b14d78982356b6bf8877e5-ab8af594": -1, "generate-tokens-c130046af74748019c65194ce8e0af97-aefc407f": -1, "generate-tokens-f35a294e2d11416c9fad24214bb8eea6-8fe5869a": -1, "generate-tokens-fdb8b5b90de9454291e8e74e7718bfb0-94cb131c": -1}}`

[Charts](index.html) | [Machine-readable results](summary.json)
