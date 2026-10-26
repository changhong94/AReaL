# R029: Work-conserving question-priority results

96 questions, 32 per batch, eight answers, concurrency16, three batches, offpolicy2;
FSDP, two NPUs, graph inference. Two reused-vLLM pairs; fixed uses answer scheduling and
shared uses question priority.

| Pair | Arm    | Training1 start/s | Training2 start/s | Loop/s | Score-slot occupancy | Mean score/s | Tokens |
| ---- | ------ | ----------------: | ----------------: | -----: | -------------------: | -----------: | -----: |
| 01   | fixed  |            185.75 |            325.80 | 518.85 |               99.59% |        2.278 | 183057 |
| 01   | latest |            157.12 |            308.58 | 557.16 |               99.21% |        2.458 | 185948 |
| 02   | fixed  |            192.98 |            320.28 | 543.32 |               99.56% |        2.396 | 178843 |
| 02   | latest |            161.70 |            322.51 | 538.70 |               99.29% |        2.372 | 181994 |

Pair 01: shared saves -38.30 s (-7.38%) vs paired fixed. Shared scoring occupancy:
99.21%; different questions overlap for 169.11 s; observed peak 4; scoring errors 0.
Completed scores needed for32/64 complete questions: \[256, 512\]. Shared-minus-fixed
decomposition: first-score start 1.38s; service work/4 34.54s; idle-equivalent 1.95s;
post-scoring tail 0.43s. This is timing accounting, not causal attribution.

Pair 02: shared saves 4.61 s (0.85%) vs paired fixed. Shared scoring occupancy: 99.29%;
different questions overlap for 181.54 s; observed peak 4; scoring errors 0. Completed
scores needed for32/64 complete questions: \[257, 512\]. Shared-minus-fixed
decomposition: first-score start 0.82s; service work/4 -4.64s; idle-equivalent 1.26s;
post-scoring tail -2.05s. This is timing accounting, not causal attribution.

Limits: these are reused-vLLM development results. The sandbox is shared, generated code
and scoring costs differ. Do not attribute every timing difference to scheduling or
claim equal training quality.

Implementation: register and return each answer independently; prefer started questions,
while spare slots accept other questions. Aggregate scoring capacity remains four. All25
CPU tests passed.

Raw logs and four chart families: `repetitions/01/` and `repetitions/02/`.

Full workspace:
`/tmp/huawei/areal-benchmarks/20261022-215001-0700_code_fsdp_c16_b3_b8d214`
