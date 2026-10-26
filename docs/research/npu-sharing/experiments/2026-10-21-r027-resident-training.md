# R027: retained training memory after a skipped inference return

## Question

When the next batch is already collected, does keeping the borrowed device in training
avoid the graph-mode switching penalty? Does preserving verified allocator caches remove
residual training overhead?

## Controlled setup

Single host, two NPUs, FSDP, Qwen2.5-1.5B-Instruct, full inference graphs, eager
training, 96 GSM8K questions, six answers per question, batch size 16, question
concurrency 16, six batches, one epoch and offpolicy 2. The shared arm retains moderated
KV rebalance and capacity-based admission. Both arms save checkpoints. Development pairs
reuse their own inference services, reset initial weights, and retain separate results.
Formal confirmation requires independent startup. Different generated answers and
packing remain confounders despite identical input questions and settings.

## Development sequence

| Candidate                         | Pair | Fixed seconds | Shared seconds | Saved seconds | Interpretation                     |
| --------------------------------- | ---: | ------------: | -------------: | ------------: | ---------------------------------- |
| Initial return decision           |    1 |        76.117 |         88.892 |       -12.775 | Large checkpoint-time outlier      |
| Initial return decision           |    2 |        74.918 |         77.776 |        -2.858 | Skip alone insufficient            |
| Resident handoff, original budget |    1 |        81.176 |         85.197 |        -4.021 | No stable improvement              |
| Resident handoff, original budget |    2 |        78.882 |         77.664 |         1.219 | Positive but mixed pair results    |
| Verified resident cache budget    |    1 |        79.485 |         76.539 |         2.946 | Shared generated 1.6% fewer tokens |
| Verified resident cache budget    |    2 |        77.972 |         73.398 |         4.574 | Shared generated 4.0% fewer tokens |

All completed shared runs above skipped five returns because the next batch was already
collected. They validate that readiness shortcut, not the accuracy of an
incomplete-batch completion predictor. Cache credit is opt-in via
`AREAL_RESIDENT_BUDGET=1`; the default-pool credit is limited to single-rank training.
Both cache-budget repetitions completed with five resident-budget hits each and no
reported OOM. Formal results will be appended after completion.

## Limits

Verified inactive storage is not a universal guarantee for arbitrary allocation shapes.
Device-budget, safety, optimizer growth and publication checks remain. Checkpoint
substage tracing does not change checkpoint contents. No slowdown result is omitted, and
wall time includes saving.

## Independent-startup confirmation, first pair

Fixed: **88.489 s**, 137,727 tokens. Shared: **80.514 s**, 137,944 tokens. Each
completed 576 responses. Shared saved **7.975 s (9.01%)** with nearly equal output
volume. Four decisions kept training because the next batch was ready; the fifth
returned to inference as a calibration probe with one question outstanding. The
private/default cache budget was hit four times. This confirms a measured positive
result for the full shared configuration, not an isolated R027 ablation or a validated
forecast formula.

Stage differences include rollout waiting (14.229 vs 10.825 s), weight publication
(13.199 vs 5.793 s), checkpoint saving (13.341 vs 10.623 s), and shared background-tail
waiting (4.325 s). Overlapping stages must not be added as serial wall time. PPO was
similar (32.253 vs 32.367 s). The independent repeat is reported below.
[Raw result](2026-10-21-r027-formal.json).

## Independent-startup confirmation, repeat

Fixed: **83.062 s**, 136,303 tokens. Shared: **75.617 s**, 135,066 tokens. Both
completed 576 responses. Shared saved **7.445 s (8.96%)**, with 0.91% fewer tokens. Five
next-batch-ready decisions skipped inference return and five resident-budget checks
preserved reusable allocations. The AReaL source tree was byte-identical across the two
formal pairs. [Raw repeat](2026-10-21-r027-formal-repeat.json).

Across both independent-startup pairs, total fixed time was 171.551 s and shared time
156.131 s: **8.99% aggregate savings**. This meets the performance objective for this
tested graph-mode two-device workload. It does not guarantee a benefit for other batch
sizes, sequence distributions, training engines, topologies or prediction-only
decisions. Both experimental options remain opt-in; this experiment did not change
global benchmark defaults.

The exact external confirmation wrapper is
`/root/areal-development/r027-formal-repeat-20261021/run.py`. It invokes the original
`run_bench.py`, with independent engine startup and `AREAL_PROFIT_AWARE_RETURN=1`,
`AREAL_RESIDENT_BUDGET=1` only for the shared arm. The earlier preparation-only failure
was an omitted wrapper default argument (`arm=None`), fixed before any NPU work; its
failed workspace is retained separately.
