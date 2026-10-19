# Benchmark comparison

[English](summary.md) | [中文](summary.zh.md)

Run: `20261015-230757-0700_math_fsdp_c16_b3_77def7`

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
- created: 20261015-230757-0700

## Main comparison

| Metric                        |   Fixed |  Shared |
| ----------------------------- | ------: | ------: |
| Training loop (s)             | 104.790 |  88.361 |
| Questions / loop second       |   0.458 |   0.543 |
| Answers / loop second         |   2.748 |   3.259 |
| Reported generated tokens     |   67501 |   68239 |
| Reported tokens / loop second | 644.155 | 772.276 |

Shared loop time reduction: **15.68%**. Speedup (fixed/shared): **1.186x**.

## Sampled device utilization

| Device metric (%) | Fixed mean | Shared mean |
| ----------------- | ---------: | ----------: |
| 1_aicore          |      12.46 |       22.30 |
| 1_hbm_used_pct    |      41.86 |       59.23 |
| 2_aicore          |       9.88 |        8.30 |
| 2_hbm_used_pct    |      44.35 |       12.45 |

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

- Fixed: `{"reconstructed": 67501, "reported": 67501, "segment_differences": {}}`
- Shared: `{"reconstructed": 68239, "reported": 68239, "segment_differences": {}}`

[Charts](index.html) | [Machine-readable results](summary.json)
