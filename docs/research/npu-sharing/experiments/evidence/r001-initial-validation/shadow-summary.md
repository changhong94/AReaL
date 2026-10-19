# Benchmark comparison

[English](summary.md) | [中文](summary.zh.md)

Run: `20261015-221301-0700_math_fsdp_c16_b3_d014c7`

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
- seed: 1
- created: 20261015-221301-0700

## Main comparison

| Metric                        |   Fixed |  Shared |
| ----------------------------- | ------: | ------: |
| Training loop (s)             | 102.026 |  89.815 |
| Questions / loop second       |   0.470 |   0.534 |
| Answers / loop second         |   2.823 |   3.207 |
| Reported generated tokens     |   67682 |   67535 |
| Reported tokens / loop second | 663.382 | 751.937 |

Shared loop time reduction: **11.97%**. Speedup (fixed/shared): **1.136x**.

## Sampled device utilization

| Device metric (%) | Fixed mean | Shared mean |
| ----------------- | ---------: | ----------: |
| 1_aicore          |      11.24 |       20.71 |
| 1_hbm_used_pct    |      41.85 |       59.03 |
| 2_aicore          |       9.67 |        7.93 |
| 2_hbm_used_pct    |      44.20 |       12.39 |

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

- Fixed: `{"reconstructed": 67682, "reported": 67682, "segment_differences": {}}`
- Shared: `{"reconstructed": 67535, "reported": 67535, "segment_differences": {}}`

[Charts](index.html) | [Machine-readable results](summary.json)
