# Benchmark comparison

[English](summary.md) | [中文](summary.zh.md)

Run: `20261015-223141-0700_math_fsdp_c16_b3_0e8e18`

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
- created: 20261015-223141-0700

## Main comparison

| Metric                        |   Fixed |  Shared |
| ----------------------------- | ------: | ------: |
| Training loop (s)             | 100.917 | 114.981 |
| Questions / loop second       |   0.476 |   0.417 |
| Answers / loop second         |   2.854 |   2.505 |
| Reported generated tokens     |   68839 |   67390 |
| Reported tokens / loop second | 682.133 | 586.095 |

Shared loop time reduction: **-13.94%**. Speedup (fixed/shared): **0.878x**.

## Sampled device utilization

| Device metric (%) | Fixed mean | Shared mean |
| ----------------- | ---------: | ----------: |
| 1_aicore          |      13.69 |       14.41 |
| 1_hbm_used_pct    |      40.99 |       58.94 |
| 2_aicore          |       9.06 |        7.81 |
| 2_hbm_used_pct    |      43.52 |       12.65 |

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

- Fixed: `{"reconstructed": 68839, "reported": 68839, "segment_differences": {}}`
- Shared:
  `{"reconstructed": 67389, "reported": 67390, "segment_differences": {"generate-tokens-59d3a4567ba849bc8cfe229388b01fb0-aa694708": -1}}`

[Charts](index.html) | [Machine-readable results](summary.json)
