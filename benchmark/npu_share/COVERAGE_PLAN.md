# Representative benchmark coverage

This is a proposed coverage plan, not a record of passed tests. Each case runs the
existing **fixed-versus-shared comparison**; do not run another fixed/shared pair
manually. The cases run sequentially and stop on the first failure.

The plan uses 16 selected configurations, not a Cartesian sweep. Start with the four
core cases, then choose the suite relevant to the changed code. This is risk-based
coverage, **not a mathematically complete pairwise covering array**.

## Run it

From any directory on the configured benchmark host, copy and execute this block:

```bash
BENCH_DIR="$(python3 - <<'PY'
import json
from pathlib import Path
p = json.loads((Path.home() / '.config/areal-benchmark/profile.json').read_text())
print(Path(p['repo']) / 'benchmark/npu_share')
PY
)"

# Inspect commands only: no data preparation, containers or NPU use.
bash "$BENCH_DIR/coverage_bench.sh" --suite core --mode plan

# Default mode: prepare inputs/configurations without training or sandbox creation.
bash "$BENCH_DIR/coverage_bench.sh" --suite core --mode prepare
```

To perform the actual comparisons after preparation succeeds:

```bash
bash "$BENCH_DIR/coverage_bench.sh" --suite core --mode run
```

Other useful selections (reuse `BENCH_DIR` from the first block):

```bash
# Investigate inference load; compare C05 against C01 and C06 against C02.
bash "$BENCH_DIR/coverage_bench.sh" --suite load --mode run

# Single-batch terminal path and off-policy boundaries.
bash "$BENCH_DIR/coverage_bench.sh" --suite boundary --mode run

# 4/8-card instance routing and TP-group readiness; ensure these devices are available.
bash "$BENCH_DIR/coverage_bench.sh" --suite topology --mode run

# A focused pair, rather than every topology.
bash "$BENCH_DIR/coverage_bench.sh" --cases 'T01 T02' --mode run

# Long-run stability; only after shorter runs pass.
bash "$BENCH_DIR/coverage_bench.sh" --suite long --mode run

# Prepare the complete representative matrix; this does NOT establish runtime support.
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode prepare
```

Use `--profile /path/to/profile.json` for a different host configuration.
`all --mode run` is supported but is deliberately not the default: it includes code
sandbox operations, 8-card runs and a 100-batch comparison.

## Selected matrix

`Offpolicy` means `--max-head-offpolicyness`. `TP` is the width of each vLLM instance.
Batch size remains 16 questions with six answers per question; each case has one epoch.
Both arms receive identical CLI values.

| ID  | Suite    | RL   | Engine   | Concurrency | Batches | NPUs |  TP | Offpolicy | Main purpose                                                                 |
| --- | -------- | ---- | -------- | ----------: | ------: | ---: | --: | --------: | ---------------------------------------------------------------------------- |
| C01 | core     | math | fsdp     |          16 |       6 |    2 |   1 |         2 | Default reference; publication and terminal cleanup                          |
| C02 | core     | math | megatron |          16 |       6 |    2 |   1 |         2 | Backend comparison against C01                                               |
| C03 | core     | code | fsdp     |          16 |       3 |    2 |   1 |         2 | E2B/Kuasar execution and reward path                                         |
| C04 | core     | code | megatron |          16 |       3 |    2 |   1 |         2 | Code workflow plus second training backend                                   |
| C05 | load     | math | fsdp     |          32 |       6 |    2 |   1 |         2 | Higher concurrency versus C01                                                |
| C06 | load     | math | megatron |          32 |       6 |    2 |   1 |         2 | Higher concurrency versus C02                                                |
| E00 | boundary | math | fsdp     |          16 |       6 |    2 |   1 |         0 | Synchronous admission; next batch executes only after weight publication     |
| E04 | boundary | math | megatron |          16 |       3 |    2 |   1 |         0 | Zero-lookahead publication on the second training backend                    |
| E01 | boundary | math | fsdp     |           1 |       1 |    2 |   1 |         1 | Minimum positive concurrency/staleness; no ordinary intermediate publication |
| E02 | boundary | math | fsdp     |          16 |       6 |    2 |   1 |         1 | Smallest positive lookahead bound versus E00 and C01                         |
| E03 | boundary | math | fsdp     |          16 |       6 |    2 |   1 |         3 | Wider lookahead versus C01; watch stale/unused work                          |
| T01 | topology | math | fsdp     |          16 |       3 |    4 |   1 |         2 | Multiple TP1 inference instances and two training ranks                      |
| T02 | topology | math | fsdp     |          16 |       3 |    4 |   2 |         2 | TP2 versus T01 with the same devices/workload                                |
| T03 | topology | math | megatron |          32 |       3 |    8 |   1 |         2 | Four training ranks; four dedicated and four borrowed TP1 instances          |
| T04 | topology | math | megatron |          32 |       3 |    8 |   2 |         2 | Two dedicated and two borrowed TP2 instances; compare with T03               |
| L01 | long     | math | fsdp     |          16 |     100 |    2 |   1 |         2 | 1,600 unique questions; repeated publication, cleanup and resource growth    |

E01 deliberately combines boundaries to cheaply find terminal-path failures; it is not a
one-variable performance ablation. Core/load/off-policy pairs and T01/T02, T03/T04 hold
other CLI settings fixed. Generated responses can nevertheless differ.

One training group always uses half the NPUs. Fixed mode uses the other half for
inference; shared mode can borrow the training half. It does not dynamically resize the
training group. The number of dedicated instances is `(NPUs / 2) / TP`.

## What each suite must establish

- **Core:** both arms finish; expected question/answer counts and identical input
  hashes; usable checkpoints, summaries and all four chart families. Phase charts have
  both English and Chinese files; other families remain English.
- **Code:** actual sandbox create/execute/delete preflight and reward execution succeed;
  global reward concurrency is matched. Preparation alone does not test the service,
  authentication or code execution. Infrastructure failures are not accepted as
  zero-reward training examples.
- **Load/off-policy:** inspect loop time, token counts, throughput, per-batch readiness,
  retained/migrated work and memory pressure. `saved` in an R label is the fixed
  interval minus shared interval, not cumulative time lead.
- **Topology:** confirm device/rank mapping, per-rank budgets, collective readiness and
  successful publication. A memory-pressure fallback is **not covered merely because
  TP>1**; use a separate controlled asymmetric-pressure test if changing offload/mapping
  code. Do not induce OOM during a live optimizer update.
- **Long:** all 100 updates complete; check whether RAM/HBM, file descriptors,
  background tasks and publication latency grow with step count. The default telemetry
  does not automatically collect every one of these resources, so a claim about leakage
  needs additional measurements, not just a successful exit.

For three batches expect 48 questions / 288 answers per arm; six batches expect 96 /
576; one batch expects 16 / 96; 100 batches expect 1,600 / 9,600. The tokenizer and
dataset preparation must confirm enough eligible distinct rows. Six answers are separate
samples, not six copies of a reference answer.

No fixed speedup threshold is a functional pass criterion. A slowdown is a finding to
investigate, not automatically an implementation failure. For performance claims, rerun
the focused pair at least three times, report every repetition and token counts, and
separate model quality from throughput. The current runner uses a fixed seed and
fixed-before-shared order: repetitions do not provide seed diversity or remove order
bias. Same-input replay is needed to isolate compute cost.

## Coverage gaps and optional follow-up

- TP4 on eight cards is optional and previously lacked end-to-end validation. First
  prepare, then do a **one-batch** check before considering longer runs:

  ```bash
  python3 "$BENCH_DIR/run_bench.py" --rl math --engine megatron \
    --npu-count 8 --inference-tp 4 --concurrency 32 --batches 1 --prepare-only
  # Remove --prepare-only only when explicitly ready to test this topology.
  ```

- This plan does not exhaust code × all topologies × all off-policy values. If changing
  sandbox routing under multiple instances, repeat T02 with `--rl code` before claiming
  that combination works.

- Concurrency 64+, larger models, different context lengths, multiple nodes, distributed
  failure recovery and training-quality evaluation are separate tests.

- Invalid values (for example concurrency 0, offpolicy -1, NPUs 3, or TP2 on two total
  NPUs) belong in CPU argument/config validation, not hardware experiments.

- A 100-batch success is not proof of indefinite stability. Report the actual scope.

## Results and failure handling

The wrapper creates `coverage-<date-time-offset>/` under the profile's persistent
`workspace_root`, outside the repository. Each child benchmark gets its normal
timestamped workspace with `settings.json`, `summary.json`, `summary.md`,
`summary.zh.md`, `index.html`, logs and figures. `C01.log`, etc. contain the
`WORKSPACE=` line; `cases.tsv` records the selected parameters and process exit code.

Preparation produces data/configuration and a `prepared` state, not performance charts
or measured speedups. If any case fails, the wrapper stops and preserves its log;
inspect the child `state.json` and logs, fix the issue, then rerun only the failed ID.
It does not automatically retry failures or recover checkpoints.

`Offpolicy=0` is a supported synchronous-admission case, not an invalid-input test. At
version v, at most (v + 1) batches can have been submitted/accepted in total; finishing
a batch does not permit the next one until the published version advances. Both cards
can still perform shared inference between training phases. Reopening inference during
PPO is skipped because no future batch can execute then. E00 and E04 cover this boundary
for FSDP and Megatron respectively.

## Recorded zero-lookahead smoke validation

On 2026-10-15, the FSDP two-NPU smoke run `20261015-224950-0700_math_fsdp_c16_b2_1f7359`
completed fixed and shared arms with two batches, 16 questions per batch, and six
answers per question. Each arm completed 192 answers: all 96 batch-one segments used
version 0 and all 96 batch-two segments used version 1. The trace audit found no version
violations. This validates the two-batch boundary path, not the full E00 six-batch case
or Megatron E04. Full logs and `offpolicy-zero-audit.json` are retained in
`/root/areal-benchmarks/` under the run ID above.
