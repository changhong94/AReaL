# Benchmark comparison

[English](summary.md) | [中文](summary.zh.md)

Run: `20261015-235533-0700_math_fsdp_c16_b3_795d4f`

## Configuration

- rl: math
- engine: fsdp
- concurrency: 16
- batches: 3
- questions: 48
- answers: 6
- npu_count: 2
- inference_tp: 1
- max_head_offpolicyness: 2
- running_rebalance: active
- seed: 1
- created: 20261015-235533-0700

## Main comparison

| Metric                        |   Fixed |  Shared |
| ----------------------------- | ------: | ------: |
| Training loop (s)             | 100.868 |  94.355 |
| Questions / loop second       |   0.476 |   0.509 |
| Answers / loop second         |   2.855 |   3.052 |
| Reported generated tokens     |   68403 |   66134 |
| Reported tokens / loop second | 678.144 | 700.906 |

Shared loop time reduction: **6.46%**. Speedup (fixed/shared): **1.069x**.

## Sampled device utilization

| Device metric (%) | Fixed mean | Shared mean |
| ----------------- | ---------: | ----------: |
| 1_aicore          |      13.45 |       19.43 |
| 1_hbm_used_pct    |      42.25 |       58.25 |
| 2_aicore          |       9.98 |        8.04 |
| 2_hbm_used_pct    |      44.31 |       12.38 |

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

- Fixed: `{"reconstructed": 68403, "reported": 68403, "segment_differences": {}}`
- Shared:
  `{"reconstructed": 66135, "reported": 66134, "segment_differences": {"generate-tokens-7b9c57c6620845898f0add4af818c68e-afe2062a": 1}}`

[Charts](index.html) | [Machine-readable results](summary.json)
