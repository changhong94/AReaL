# Representative benchmark coverage

[中文版](COVERAGE_PLAN.zh.md)

This is a proposed coverage plan, not a record of passed tests. Each case runs the
existing **fixed-versus-shared comparison**; do not run another fixed/shared pair
manually. The cases run sequentially and continue after individual failures. The suite
returns a nonzero exit code if any case failed.

The plan uses 28 selected configurations, not a Cartesian sweep. Start with the four
core cases, then choose the suite relevant to the changed code. This is risk-based
coverage, **not a mathematically complete pairwise covering array**.

## Run it

From any directory on the configured benchmark host, define `BENCH_DIR`. Optionally
inspect commands with `plan`, then execute `run` directly. Do not run `prepare` first:
`run` performs its own preparation.

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
```

To perform the actual comparisons, run the following command directly. It includes its
own preparation:

```bash
bash "$BENCH_DIR/coverage_bench.sh" --suite core --mode run
```

Each `prepare` or `run` invocation creates a new timestamped coverage directory. `run`
does **not** resume or reuse a previous `prepare` directory: it freezes the current
source and prepares its own inputs. Running both therefore creates two directories, but
trains only during `run`. `plan` creates no directory. To avoid redundant
preparation/output directories, skip `prepare` when a separate preflight is unnecessary.

Other useful selections (reuse `BENCH_DIR` from the first block):

```bash
# Investigate inference load; compare C05 against C01 and C06 against C02.
bash "$BENCH_DIR/coverage_bench.sh" --suite load --mode run

# Single-batch terminal path and off-policy boundaries.
bash "$BENCH_DIR/coverage_bench.sh" --suite boundary --mode run

# 4/8-card instance routing and TP-group readiness; ensure these devices are available.
bash "$BENCH_DIR/coverage_bench.sh" --suite topology --mode run

# Unequal inference:training ratios, including TP2 and Megatron.
bash "$BENCH_DIR/coverage_bench.sh" --suite ratio --mode run

# A focused pair, rather than every topology.
bash "$BENCH_DIR/coverage_bench.sh" --cases 'T01 T02' --mode run

# Workload size and answer-group coverage (six selected combinations).
bash "$BENCH_DIR/coverage_bench.sh" --suite workload --mode run

# Long-run stability; only after shorter runs pass.
bash "$BENCH_DIR/coverage_bench.sh" --suite long --mode run

# Inspect the complete representative matrix without creating output directories.
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode plan
```

Use `--profile /path/to/profile.json` for a different host configuration.
`all --mode run` is supported but is deliberately not the default: it includes code
sandbox operations, 8-card runs and a 100-batch comparison.

## Selected matrix

`Offpolicy` means `--max-head-offpolicyness`. `TP` is the width of each vLLM instance.
Default batch size is 32 questions with eight answers per question; each case has one
epoch. The workload suite (C01 and W01–W05, listed in the main table below) varies batch
size over 16/32/64 and answers over 4/8/16. The core suite alone keeps 32/8 fixed; use
`--suite workload` or `--suite all` to include the variations. Both arms receive
identical CLI values. All cases use inference graph mode except G01/G02, which use
eager; training execution is unchanged.

| ID  | Suite     | RL   | Engine   | Inference mode | Concurrency | Batches | NPUs | Inference:training |  TP | Offpolicy | Batch size | Answers | Main purpose                                                                         |
| --- | --------- | ---- | -------- | -------------- | ----------: | ------: | ---: | ------------------ | --: | --------: | ---------: | ------: | ------------------------------------------------------------------------------------ |
| C01 | core      | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         32 |       8 | Default reference; publication and terminal cleanup                                  |
| C02 | core      | math | megatron | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         32 |       8 | Backend comparison against C01                                                       |
| C03 | core      | code | fsdp     | graph          |          16 |       3 |    2 | 1:1                |   1 |         2 |         32 |       8 | E2B/Kuasar execution and reward path                                                 |
| C04 | core      | code | megatron | graph          |          16 |       3 |    2 | 1:1                |   1 |         2 |         32 |       8 | Code workflow plus second training backend                                           |
| C05 | load      | math | fsdp     | graph          |          32 |       6 |    2 | 1:1                |   1 |         2 |         32 |       8 | Higher concurrency versus C01                                                        |
| C06 | load      | math | megatron | graph          |          32 |       6 |    2 | 1:1                |   1 |         2 |         32 |       8 | Higher concurrency versus C02                                                        |
| E00 | boundary  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         0 |         32 |       8 | Synchronous admission; next batch executes only after weight publication             |
| E04 | boundary  | math | megatron | graph          |          16 |       3 |    2 | 1:1                |   1 |         0 |         32 |       8 | Zero-lookahead publication on the second training backend                            |
| E01 | boundary  | math | fsdp     | graph          |           1 |       1 |    2 | 1:1                |   1 |         1 |         32 |       8 | Minimum positive concurrency/staleness; no ordinary intermediate publication         |
| E02 | boundary  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         1 |         32 |       8 | Smallest positive lookahead bound versus E00 and C01                                 |
| E03 | boundary  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         3 |         32 |       8 | Wider lookahead versus C01; watch stale/unused work                                  |
| T01 | topology  | math | fsdp     | graph          |          16 |       3 |    4 | 1:1                |   1 |         2 |         32 |       8 | Multiple TP1 inference instances and two training ranks                              |
| T02 | topology  | math | fsdp     | graph          |          16 |       3 |    4 | 1:1                |   2 |         2 |         32 |       8 | TP2 versus T01 with the same devices/workload                                        |
| T03 | topology  | math | megatron | graph          |          32 |       3 |    8 | 1:1                |   1 |         2 |         32 |       8 | Four training ranks; four dedicated and four borrowed TP1 instances                  |
| T04 | topology  | math | megatron | graph          |          32 |       3 |    8 | 1:1                |   2 |         2 |         32 |       8 | Two dedicated and two borrowed TP2 instances; compare with T03                       |
| R01 | ratio     | math | fsdp     | graph          |          16 |       3 |    3 | 2:1                |   1 |         2 |         32 |       8 | 2:1 allocation; 2 dedicated and one borrowed TP1 instances                           |
| R02 | ratio     | math | fsdp     | graph          |          16 |       3 |    4 | 3:1                |   1 |         2 |         32 |       8 | 3:1 allocation; 3 dedicated and one borrowed TP1 instances                           |
| R03 | ratio     | math | megatron | graph          |          16 |       3 |    6 | 2:1                |   2 |         2 |         32 |       8 | 2:1 allocation; 2 dedicated and one borrowed TP2 instances                           |
| R04 | ratio     | math | megatron | graph          |          16 |       3 |    8 | 3:1                |   2 |         2 |         32 |       8 | 3:1 allocation; 3 dedicated and one borrowed TP2 instances                           |
| G01 | execution | math | fsdp     | eager          |          16 |       6 |    2 | 1:1                |   1 |         2 |         32 |       8 | Eager control matched to graph C01                                                   |
| G02 | execution | math | megatron | eager          |          16 |       6 |    2 | 1:1                |   1 |         2 |         32 |       8 | Eager control matched to graph C02                                                   |
| W01 | workload  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         16 |       8 | Smaller question batch versus C01                                                    |
| W02 | workload  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         64 |       8 | Larger question batch versus C01                                                     |
| W03 | workload  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         32 |       4 | Smaller answer group versus C01                                                      |
| W04 | workload  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         32 |      16 | Larger answer group versus C01                                                       |
| W05 | workload  | math | fsdp     | graph          |          16 |       6 |    2 | 1:1                |   1 |         2 |         64 |      16 | Combined high-load corner                                                            |
| L01 | long      | math | fsdp     | graph          |          16 |     100 |    2 | 1:1                |   1 |         2 |         32 |       8 | 3,200 unique questions; repeated publication, cleanup and resource growth            |
| L02 | long      | math | megatron | graph          |          32 |     100 |    2 | 1:1                |   1 |         2 |         32 |       8 | Megatron long-run stability under higher concurrency; inference:training remains 1:1 |

E01 deliberately combines boundaries to cheaply find terminal-path failures; it is not a
one-variable performance ablation. Core/load/off-policy pairs and T01/T02, T03/T04 hold
other CLI settings fixed. Generated responses can nevertheless differ.

L01 and L02 both use two NPUs with inference:training `1:1`. L02 changes both the
training backend and concurrency; it is complementary stability coverage, not a
single-variable performance ablation. Each arm processes 3,200 questions and 25,600
answers across 100 batches.

The unified matrix includes R01–R04 for unequal allocation ratios. All other cases use
`--inference-training-ratio 1:1`.

Each ratio case uses math RL and one distributed training group. Fixed mode uses only
dedicated inference devices; shared mode may also borrow training devices. Both groups
must be divisible by TP. The training group is not dynamically resized. R01/R03 and
R02/R04 exercise equivalent instance counts with different TP widths and training
backends; they are functional coverage, not single-variable ablations.

All cases explicitly use `--adaptive-concurrency capacity` for the shared arm. The fixed
arm retains its requested baseline concurrency. R025 scales by ready instances relative
to dedicated instances: 2:1 can reach 1.5 times baseline, and 3:1 can reach 4/3 times
baseline, subject to readiness, queue and off-policy gates. These cases are a test plan,
not a claim of validated hardware performance.

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
- **Ratio:** verify dedicated/borrowed rank sets, training-device ownership, publication
  to every dedicated instance, and R025 targets for unequal pools.
- **Long:** all 100 updates complete; check whether RAM/HBM, file descriptors,
  background tasks and publication latency grow with step count. The default telemetry
  does not automatically collect every one of these resources, so a claim about leakage
  needs additional measurements, not just a successful exit.

Expected questions per arm equal `batches × batch_size`; expected answers equal
`batches × batch_size × answers`. With the default 32/8 workload, three batches mean 96
questions / 768 answers; six mean 192 / 1,536; one means 32 / 256; 100 mean 3,200 /
25,600. W01–W05 use the explicit values in the unified matrix. The tokenizer and dataset
preparation must confirm enough eligible distinct rows. Multiple answers are separate
samples, not copies of a reference answer.

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

- Invalid values (for example concurrency 0, offpolicy -1, NPUs 3 with ratio 1:1, or TP2
  on two total NPUs) belong in CPU argument/config validation, not hardware experiments.

- A 100-batch success is not proof of indefinite stability. Report the actual scope.

## Results and failure handling

The wrapper creates `coverage-<date-time-offset>/` under the profile's persistent
`workspace_root`, outside the repository. Each child benchmark gets its normal
timestamped workspace with `settings.json`, `summary.json`, `summary.md`,
`summary.zh.md`, `index.html`, logs and figures. `C01.log`, etc. contain the
`WORKSPACE=` line; `cases.tsv` records the selected parameters and process exit code.

Preparation produces data/configuration and a `prepared` state, not performance charts
or measured speedups. If any case fails, the wrapper preserves its log and continues
with the remaining cases. After all selected cases have been attempted, it returns exit
code 1 if any failed, otherwise 0. Inspect the child `state.json` and logs, fix the
issue, then rerun only the failed ID. It does not automatically retry failures or
recover checkpoints.

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

## Inference execution mode

`--inference-mode graph` is the benchmark default and selects vLLM Ascend `FULL`,
covering prefill/mixed batches and decode; training execution is unchanged. Capture
buckets include the scheduler token limit (2048 by default), rather than only small
decode batches. The worker logs `AREAL_GRAPH_PREFILL_MIXED` and `AREAL_GRAPH_DECODE`
with the actual dispatch mode once per phase/mode. Any eager fallback is therefore
visible as `mode=NONE`. Use `--inference-mode eager` for the original execution path.
Both fixed and shared arms receive the same mode, persisted as `inference_mode` in
settings.

Coverage defaults to graph for the existing workload/topology/ratio/long cases.
`--suite execution` selects C01/C02 graph and G01/G02 eager controls.
`coverage_bench.sh --inference-mode eager` overrides all selected cases.

Shared graph support is scoped to the existing Qwen2 BF16 full-attention, uni/mp
equal-TP replicas. Sparse KV keeps stable virtual addresses; graph pools remain resident
outside reclaimable storage. Captured weight/buffer storage must not change during
weight publication. Unsupported graph modes or storage replacement fail explicitly
instead of silently replaying stale addresses.

## Batch-size and answer-group coverage

Run `--suite workload` to compare C01 with five additional cases. All use math, FSDP,
graph inference, concurrency 16, six batches, two NPUs, TP1, off-policy 2, and
inference:training 1:1. Four cases change one workload factor; W05 tests the combined
high-load corner. This is not a nine-combination Cartesian sweep.

The unified matrix above lists all workload cases. Answers per batch equal
`Batch size × Answers`; questions per run equal `Batches × Batch size`. Each command
passes both values explicitly, and `cases.tsv` records them. These cases compare system
performance; they do not establish which workload shape gives the best RL learning
quality. Existing historical results retain their recorded settings and are not
reinterpreted using the new defaults.

R027 profit-aware inference return and verified resident-cache budgeting are enabled by
default for the shared arm (`AREAL_PROFIT_AWARE_RETURN=1`, `AREAL_RESIDENT_BUDGET=1`).
The fixed arm explicitly disables both. Coverage cases inherit these defaults. Cache
credit retains existing safety and topology checks; default-pool credit is limited to
single-rank FSDP.

### Coverage overview pages

The timestamped coverage directory contains `index.html` (English), `index.zh.html`
(Chinese), `summary.md`, `summary.zh.md`, and machine-readable `summary.json`. Reports
are updated before and after each case and at normal exit or handled interruption. They
show the selected case parameters, status, fixed/shared loop seconds, time reduction,
token counts and links to the individual charts, summary, state and log. Initialization
is excluded from loop time. Missing measurements remain unavailable; failed cases do not
receive a performance claim. Forced termination such as SIGKILL cannot execute the final
report hook.

To rebuild an existing suite summary without running experiments:

```bash
python3 "$BENCH_DIR/coverage_summary.py" /path/to/coverage-directory --final
```

For older coverage directories without `planned-cases.tsv`, only recorded cases can be
recovered. The report does not invent missing planned cases. Omit `--final` when
inspecting a suite that is still running.

### Pause, resume, and manually rerun cases

For an already running suite, identify the **Bash coverage_bench.sh PID**, not its
driver, run_bench.py, or training process. Control only that launcher:

```bash
python3 "$BENCH_DIR/coverage_control.py" pause --pid <coverage-bash-pid>
python3 "$BENCH_DIR/coverage_control.py" status --pid <coverage-bash-pid>
# Wait for status=paused, then optionally rerun selected cases:
python3 "$BENCH_DIR/coverage_control.py" rerun --pid <coverage-bash-pid> --cases 'T01 T02'
# Continue the original suite in its original workspace:
python3 "$BENCH_DIR/coverage_control.py" resume --pid <coverage-bash-pid>
```

`pause_pending` lets the current comparison finish normally, including cleanup; `paused`
means the launcher has no live direct children. Verify container cleanup before reusing
devices, especially after a failed case. The completed case's TSV entry and suite
summary may only be finalized when the launcher resumes. No process-group stop is used.
This is a live-process pause, not restart recovery: the remote host and launcher must
remain alive. Laptop disconnection is safe. Add `--wait` to the pause command to wait
until its active children finish.

Manual reruns retain original results and create a separate timestamped coverage
workspace. They use the current source/case matrix and the original launcher's profile
and inference-mode override. They do not replace original failures or change the paused
suite's queue. Historical configurations should be checked against planned-cases.tsv if
the matrix changed. Without a live paused suite, use the existing standalone command:

```bash
bash "$BENCH_DIR/coverage_bench.sh" --mode run --cases 'T01 T02' --profile /path/to/profile.json
```

### Random case order

Use `--shuffle` to randomize selected cases once. `--shuffle-seed 42` enables
reproducible ordering. Without either option, the original order remains. The seed and
order are printed and saved to `case-order.json`; `planned-cases.tsv` uses that order. A
paused live suite retains its order on resume. Individual failures still allow
subsequent cases to run.

```bash
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode run --shuffle
bash "$BENCH_DIR/coverage_bench.sh" --suite all --mode plan --shuffle-seed 42
```
