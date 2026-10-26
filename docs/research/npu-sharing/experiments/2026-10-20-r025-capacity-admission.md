# R025 capacity-adaptive admission: development validation

Date: 2026-10-20 (America/Vancouver). Status: positive exploratory evidence;
independent-start confirmation pending.

Workspace: `/root/areal-development/20261020-102453-0700_math_fsdp_c16_b6_590372`.

## Protocol

Two NPUs, FSDP math RL, 96 questions, six answers per question, baseline concurrency 16,
six batches, offpolicy 2, same engine hard limits and tracing. Four repetitions each ran
fixed then shared. Shared modes alternated off/capacity/off/capacity. vLLM services were
reused; immutable initial DRAM weights were restored and checked before each run.
Training processes and optimizer state were recreated. R001 remained enabled with the
same accepted policy. Inputs and source were frozen in the workspace.

## Results

| Pair | Shared static 16 (s) | Shared R025 (s) | Reduction | Static tokens | R025 tokens |
| ---- | -------------------: | --------------: | --------: | ------------: | ----------: |
| 1    |               179.51 |          157.22 |    12.42% |        137362 |      136557 |
| 2    |               182.65 |          149.18 |    18.32% |        138310 |      134362 |

Mean shared-loop time decreased from **181.08 s to 153.20 s** (**15.40%**). Logical
output tokens changed by **-1.72%**; aggregate logical-token throughput rose from
**761.19 to 884.22 tokens/s**. Each run completed 576 answers. Times exclude
initialization, reset and rendering.

Actual overlapping question sessions peaked at 16 in both static runs and 32 in both
adaptive runs. Target traces visited 16/20/24/28/32 and fell back to 16 when capacity
was withdrawn (23 and 25 changes). Mean sequences per pure-decode step rose from
31.96/33.21 to 40.32/41.93; these are step averages, not time-weighted utilization. The
native hard sequence limit remained 64. Tokens above come from accepted rollout seqlen
minus prompt length; native replay/trace counters can have different scope.

| Repetition | Fixed (s) | Shared (s) | R025 mode | Shared KV imports |
| ---------- | --------: | ---------: | --------- | ----------------: |
| 01         |    194.53 |     179.51 | off       |                 8 |
| 02         |    230.52 |     157.22 | capacity  |                 0 |
| 03         |    214.24 |     182.65 | off       |                 9 |
| 04         |    195.97 |     149.18 | capacity  |                 5 |

## Interpretation and limitations

The results support the underfilled-replica hypothesis for this workload: more question
admission increased actual overlap and native decode batch size. This does not prove
universal benefit or unique advantage over static concurrency 32. A static-32 control,
independent-start confirmation, other workloads and TP>1 validation are still pending.
Two trials per mode are not a confidence interval. Generated answers, lengths and
migration outcomes differ; no equal-quality claim is made. The first adaptive run had
zero confirmed KV imports, the second five; all shared runs retained the same policy.
Fixed controls vary from 194.53 to 230.52 seconds, showing significant tail variability.

The implemented policy is readiness/capacity-driven, not full throughput feedback. The
reusable-tool cleanup repair is engineering support, not a research contribution. 87
targeted regression cases passed across the test runs; formatting/lint passed.

[Machine-readable evidence](2026-10-20-r025-capacity-admission.json)
