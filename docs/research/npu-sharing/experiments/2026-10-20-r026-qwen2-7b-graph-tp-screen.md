# R026: Graph-mode TP screening — 2026-10-20

Qwen2-7B-Instruct, BF16, Ascend 910B2. FULL_DECODE_ONLY reverses the eager-mode TP
result: TP2 reduces warmed completion time by 15.21–28.57% versus TP1 in these
workloads.

| Workload               | Eager TP1 (s) | Eager TP2 (s) | Graph TP1 (s) | Graph TP2 (s) | Graph TP2 time reduction |
| ---------------------- | ------------: | ------------: | ------------: | ------------: | -----------------------: |
| historical_tail_group  |        16.668 |        19.537 |         4.406 |         3.736 |                   15.21% |
| historical_last_single |        15.466 |        18.130 |         3.847 |         2.748 |                   28.57% |
| fixed512_c1            |        28.615 |        33.740 |         7.140 |         5.313 |                   25.59% |
| fixed512_c4            |        31.152 |        35.253 |         7.405 |         5.511 |                   25.58% |
| fixed512_c16           |        31.295 |        36.095 |         8.332 |         6.438 |                   22.74% |
| fixed512_c32           |        31.994 |        36.574 |         9.165 |         7.697 |                   16.02% |

Method: identical image, model, token inputs and fixed output lengths as the eager
screening. Independent TP1a → TP2 → TP1b engines; two warmups and three measurements per
case per arm. TP1 pools six measurements; TP2 uses three. Initialization and graph
capture are excluded. Logs confirm graph capture. No training, live KV migration, or
weight updates are tested. This is one screening sequence, not a confidence interval or
a production guarantee. Eager comparison is from the earlier run, not interleaved. Graph
mode may enable multiple compiler optimizations; no profiler attribution was performed.
Request metrics were disabled, so these are completion times, not measured TTFT/TPOT.

Compatibility: the current shared KV slab adapter explicitly requires eager mode; this
standalone vLLM result does not validate shared RL graph support. TP2 uses twice the
devices, so lower latency does not imply lower NPU-seconds. R026 must amortize
migration/import/resharding and any engine preparation cost.

Initialization seconds: {"tp1a": 112.20610621571541, "tp2": 118.77005526423454, "tp1b":
112.18364439532161}

Evidence: [comparison](evidence/r026-graph-20261020/comparison.json),
[plan](evidence/r026-graph-20261020/plan.json).
