# Multi-NPU RL comparison runner

Run from the repository root on the NPU host. This tool compares fixed and shared NPU
execution using one frozen source snapshot. The default uses two NPUs; larger topologies
use one training group and one or more inference instances. The tool does not modify or
commit the source repository.

```bash
python3 benchmark/npu_share/run_bench.py
python3 benchmark/npu_share/run_bench.py --rl math --concurrency 32 --engine megatron --batches 3
python3 benchmark/npu_share/run_bench.py --rl code --concurrency 16 --engine fsdp --batches 6
python3 benchmark/npu_share/run_bench.py --rl math --batches 100
```

| Option                     | Default                  | Meaning                                                                                                |
| -------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------ |
| `--rl`                     | `math`                   | `math` (GSM8K) or `code` (LeetCode through E2B/Kuasar)                                                 |
| `--concurrency`            | `16`                     | Concurrent questions across the rollout controller                                                     |
| `--engine`                 | `fsdp`                   | `fsdp` or `megatron`; the profile selects the interpreter/image                                        |
| `--inference-tp`           | `1`                      | Devices per vLLM instance; must divide half the NPU count                                              |
| `--npu-count`              | `2`                      | Even total NPU count; supports 2, 4, or 8                                                              |
| `--max-head-offpolicyness` | `2`                      | Rollout head off-policy limit, identical in both arms; nonnegative integer; 0 disables stale lookahead |
| `--batches`                | `6`                      | Completed training batches in one epoch, in each arm                                                   |
| `--workspace-root`         | Profile setting          | Output root, which must be outside the source repository                                               |
| `--timeout`                | `max(3600, batches*600)` | Seconds allowed for each arm                                                                           |
| `--prepare-only`           | Off                      | Prepare data/configuration without NPU use or sandbox creation                                         |
| `--render-only PATH`       | None                     | Regenerate charts from a completed workspace; no NPU use                                               |

Batch size remains **16 questions**, with **six answers per question**. Concurrency does
not change batch size. Six batches require 96 distinct questions; 100 batches require
1,600. The runner selects eligible rows from the full local datasets, validates them
with the actual tokenizer/dataset loader, and refuses insufficient data instead of
silently repeating questions. The initial model, seed, dataset, logging and native
scheduler limit are matched between arms. Generated answers and batch membership can
still differ.

The native `max_num_seqs` limit is `max(64, 4*concurrency)` per vLLM instance. The
controller queue size is at least 32, including for concurrency 1. Code reward execution
has a separate aggregate limit of four sandbox operations across all rollout processes;
this does not reduce six answers to four. Code RL requires the configured Kuasar service
to be available and passes create/execute/delete preflight before acquiring NPU
resources. Infrastructure failures abort the run instead of becoming zero rewards.

## Output

The first output line prints `WORKSPACE=...`. The installed host profile uses:

```text
<workspace_root>/YYYYMMDD-HHMMSS-0700_math_fsdp_c16_b6_<id>/
  settings.json           # Exact parameters, image ID, timezone
  source/                 # Identical frozen source for both arms
  source-manifest.json    # Source hashes and base commit
  data/                   # Selected input data
  data-manifest.json      # Source data hashes and selected rows
  fixed/                  # Fixed-mode logs, metrics, traces and checkpoints
  latest/                 # Shared-mode logs, metrics, traces and checkpoints
  state.json              # Current stage, failure details and cleanup status
  summary.json            # Timing comparison and measurement audit
  summary.md              # English metrics, speedup and measurement scope
  summary.zh.md           # Chinese report from the same measurements
  index.html              # Clickable index of all generated PNGs
  figures/                # Four chart families, with detail pages for large runs
```

Timestamp timezone defaults to America/Vancouver in this host profile; its UTC offset
changes with daylight saving time. `/tmp` results may disappear after a reboot. Use
`--workspace-root /path/outside/repo` for a persistent filesystem.

Four chart families are generated automatically:

1. Fixed/shared phase comparison, with start/end/duration labels. Long runs also receive
   six-batch detail pages.
1. Per-request timelines for each arm, grouped by question, one full image per arm for
   up to six batches, or 16 questions per page for larger runs, with all inference
   instances, queue/execution shading and interruption markers.
1. Sampled AI Core and HBM utilization for every selected device and both arms.
1. Per-second token output, colored by weight version.

Stars mark the last consumed answer in each batch; vertical lines mark actual controller
batch readiness. For code RL, sandbox rewards can introduce additional delay between
answer completion and batch readiness. Token bars reconstruct single-token
execution-window ends, not exact emission timestamps. Count discrepancies are retained
in `summary.json`. HBM includes reservations/caches; approximately two-second
utilization sampling can miss brief peaks.

## Disconnecting and monitoring

Keep the launcher running on the remote host, not the laptop:

```bash
nohup python3 benchmark/npu_share/run_bench.py --engine megatron --batches 6 \
  > /tmp/areal-benchmark-launch.log 2>&1 < /dev/null &
tail -f /tmp/areal-benchmark-launch.log
# Read WORKSPACE from that log, then inspect WORKSPACE/state.json.
```

The runner holds a host-wide lock to reject overlapping benchmark launchers. It checks
for other NPU users and stops only its own experiment container on failure or
interruption. Arrange available NPU resources manually before starting. SIGINT/SIGTERM
trigger cleanup; SIGKILL, host shutdown and a Docker outage cannot guarantee cleanup.

To rerender after a plotting failure, without repeating training:

```bash
python3 benchmark/npu_share/run_bench.py --render-only /absolute/workspace/path
```

## Host configuration

Host-specific paths live in `~/.config/areal-benchmark/profile.json`, outside the source
repository. Use `--profile` to select another profile. The installed profile defaults to
physical NPUs 1 and 2 on the existing 910B host; see NPU topology for larger
allocations. The selected image must already exist; the tool never rebuilds a driver or
image.

Credentials remain in the existing Kuasar configuration directory, mounted read-only.
They are not copied into the workspace or included in its manifest. The isolated
plotting interpreter is specified by `plot_python`; its dependencies are Matplotlib,
NumPy and Pillow. Training continues to use each engine's existing interpreter,
including Python 3.12 for Megatron.

The tool enables device/request tracing in both arms because request and token charts
require it. These measurements include tracing overhead. This is a performance
comparison, not a held-out model-quality evaluation. The runner does not claim that
fixed mode is pristine upstream code.

## CPU-only checks

```bash
python3 benchmark/npu_share/run_bench.py --help
python3 -m unittest discover -s benchmark/npu_share -p test_runner.py
```

## Console progress

The runner prints timestamped stage changes, model initialization completion, completed
training batches, failures, and output locations. Output is flushed immediately,
including when redirected to a file. After 60 seconds without a new message, a waiting
notice reports the current stage and elapsed time. This notice confirms that the
supervisor is alive; it does not prove that a worker is making progress. Detailed
training output stays in the logs; the console also prints a `docker logs -f` command
for the active experiment container. Batch progress reads existing phase records every
five seconds without device synchronization.

## NPU topology

`--npu-count` supports 2 (default), 4, or 8 devices. Half form **one logical training
group**, using pure data parallelism to update one model. The other half are dedicated
inference devices. Shared mode additionally borrows the training devices for inference
when per-device budgets permit it.

`--inference-tp` defaults to 1 and must divide half the device count. Each vLLM instance
uses this many tensor-parallel devices. Requests are routed between instances, not
between the members of one TP instance.

| Total NPUs | Inference TP | Training devices | Dedicated instances | Maximum shared-mode instances |
| ---------- | ------------ | ---------------- | ------------------- | ----------------------------- |
| 2          | 1            | 1                | 1                   | 2                             |
| 4          | 1            | 2                | 2                   | 4                             |
| 4          | 2            | 2                | 1                   | 2                             |
| 8          | 1            | 4                | 4                   | 8                             |
| 8          | 2            | 4                | 2                   | 4                             |
| 8          | 4            | 4                | 1                   | 2                             |

```bash
python3 benchmark/npu_share/run_bench.py --npu-count 4 --batches 3
python3 benchmark/npu_share/run_bench.py --npu-count 8 --inference-tp 2 --engine megatron
python3 benchmark/npu_share/run_bench.py --npu-count 8 --inference-tp 4
```

The host profile's `devices` preserves the original two-device selection.
`available_devices` optionally supplies the ordered larger-device inventory; on the
current eight-device host it defaults to indices 0 through 7. The selected IDs and
allocations are printed and retained in `settings.json`.

Memory decisions are made for each device separately. A TP member under pressure may
preserve KV pages in host memory while other members retain their mappings. The instance
resumes only when all members acknowledge the required mappings and the shared scheduler
uses the minimum feasible block capacity. Dynamic growth performs coordinated CPU
preflight and mapping acknowledgements; these checks have overhead. TP>1 keeps paused
requests on their original instance during the training handoff; TP=1 retains the
migration strategy. Same-version execution may overlap training only after all
per-device budgets pass. Weight replacement still waits for outstanding old-version
execution to drain or be interrupted.

Request and token charts use inference instances (with physical device IDs in the
labels); TP tokens are counted once. Utilization charts show each physical device. Phase
charts summarize the training/shared and dedicated groups.

## Multi-device validation (2026-10-15)

The targeted regression suite passed 212 tests, including per-device reclamation, TP
failure propagation, grouped worker launch/cleanup, live KV capacity changes, and
allocator conservation. The runner's five CPU orchestration tests also pass.

On Ascend 910B with Qwen2.5-1.5B-Instruct:

- Four devices, FSDP, inference TP=1: both fixed and shared arms completed two batches.
  All four chart families rendered successfully.
- Eight devices, Megatron, inference TP=2: fixed and shared arms each completed two
  batches (32 questions, six answers per question). The shared arm was retried after
  implementation fixes, so these runs are functional validation, not a matched
  performance comparison.
- Shared TP execution overlapped the first PPO update: each borrowed TP instance
  recorded 44 synchronized model-step windows overlapping that update. The final update
  had no remaining inference work.
- A native two-device asymmetric-residency check moved only 4 MiB of KV from one device
  to host RAM, retained the other device's mappings, then restored and verified every
  byte.

TP=4 passed configuration validation but has not yet received an end-to-end hardware
run. TP coordination currently adds CPU collective checks to decoding; these checks
prioritize correctness and may affect performance.

Shared Ascend vLLM processes default `HCCL_NPU_SOCKET_PORT_RANGE` to `auto` to avoid
colliding with the training process's device-side listener. Explicit user values are
preserved. See the
[CANN environment-variable reference](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0144.html)
for hardware and communication-operator constraints.

## Reading batch timing

Shared readiness labels additionally show `saved +/-Ys`, calculated as the fixed
adjacent-node interval minus the shared adjacent-node interval for the same batch index.
Both displayed seconds therefore use the same interval basis. Positive means time saved
in that interval; negative means extra time (no saving).

Readiness labels use `Rn, Xs`, where `Xs = Rn - R(n-1)`. R1 is relative to training-loop
start. This interval is not acquisition waiting time or the full lifetime of the
requests in that batch.

Phase charts label `Bn, duration` from the main-loop batch acquisition entry to the
corresponding step-tail completion, including any background tail. Shared-mode intervals
can overlap and must not be summed as wall time. `Rn` and dotted lines mark when the
controller receives the complete batch; stars mark the last consumed answer completion,
which may precede acquisition because of asynchronous prefetch. The renderer records
these boundaries in `summary.json` under `batch_step_timing`.

Both `summary.md` and `summary.zh.md` report the same measurements. Missing historical
settings are shown as unavailable rather than inferred from current defaults.

Phase charts additionally have a Chinese sibling (`overview.zh.png` and
`batches_*.zh.png`), rendered from the same figure and numeric data. The original
English file is retained. Other chart families remain English-only. An installed CJK
font such as Droid Sans Fallback or Noto Sans CJK is required for Chinese output.

## Representative coverage

See [COVERAGE_PLAN.md](COVERAGE_PLAN.md) for 14 selected configurations and copy-and-run
commands. `coverage_bench.sh` runs selected cases sequentially; its default is
preparation only. Use `--mode plan` to inspect commands or `--mode run` to execute
actual fixed/shared comparisons.

The configured host now stores workspaces under `/root/areal-benchmarks`, model and math
inputs under `/root/areal-assets`, and code inputs/dependencies under
`/root/areal-benchmark-assets/code-three-preflight`. Paths remain profile settings, not
script constants. Old `/tmp/huawei` locations are compatibility symlinks; the profile
uses persistent paths directly and does not depend on those symlinks.

## Experimental running-request rebalancing (R001)

`--running-rebalance off|shadow|budget|active|kv` controls only the shared arm and
defaults to `off`. The fixed arm always disables it. `shadow` records causal forecasts
without moving running requests. `active` may abort one eligible running request and
resume its accumulated prefix on a compatible shared replica. This rebuilds KV through
prefill; it does not transfer KV buffers.

```bash
python3 benchmark/npu_share/run_bench.py --batches 3 --running-rebalance shadow
python3 benchmark/npu_share/run_bench.py --batches 3 --running-rebalance active
python3 benchmark/npu_share/analyze_running_rebalance.py /path/to/workspace/latest
```

The policy prioritizes queued requests, requires an admitted inference-only KV budget,
checks logical KV headroom, rejects imminent training, includes estimated prefix replay
in its critical-tail forecast, and moves a request at most once. A handoff pins
source/destination epochs until ownership transfers. Rejected admission retains the
version constraint, with at most three attempts; ambiguous execution or loss of a
compatible epoch fails closed. Publication waits at most 10 seconds for the ownership
handshake; this is a failure watchdog, not a drain policy or a tunable performance
target.

This is experimental: the three-batch causal calibration estimates decode and prefill
costs, but does not yet provide a calibrated future-training deadline or contention
uncertainty model. A same-version handoff does not promise a single policy version for
the entire RL trajectory. Normal later weight publication retains the framework's
existing segment-version semantics. Sampling after prefix replay need not be bitwise
identical to uninterrupted sampling.

Use `--running-rebalance budget` as the matching-budget ablation: it establishes exactly
the active policy's inference-only KV cap and records predictions, but never aborts
running requests. Compare `budget` with `active` to isolate migration; comparing `off`
with `active` measures the combined budget and scheduling change. The `shadow` mode
remains read-only and does not change the original memory budget.

## Experimental KV transport probe and offline batch planning

These tools do not enable live vLLM KV migration. In a prepared Ascend runtime with two
available NPUs, test real-model cache transfer and continuation:

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmark/npu_share/kv_transfer_probe.py \
  --model /path/to/local/model --output /path/to/workspace/kv-probe
```

The initial scope is TP=1 Qwen2.5-1.5B BF16 on two devices. The probe uses Transformers
eager execution, not the vLLM scheduler. It records per-rank packing, HCCL and unpacking
timings and 16-token greedy continuations. Verify both rank files before interpreting
timings. Source and target use the same read-only model directory. The process-group
communicator is warmed first.

Replay a saved R001 arm's snapshots without using NPUs:

```bash
PYTHONPATH=. python3 benchmark/npu_share/replay_batch_rebalance.py \
  /path/to/benchmark/latest --probe /path/to/workspace/kv-probe \
  --output /path/to/workspace/batch-replay.json
```

The report distinguishes replay from hypothetical KV moves and varies the unmeasured
ownership/import cost. It includes planning overhead and does not authorize execution.
HCCL tensor transfer correctness does not establish vLLM block-ownership import
correctness or end-to-end RL speedup.

## Experimental live KV handoff

`--running-rebalance kv` enables the two-device, TP=1 live KV adapter for the shared
arm. The fixed arm remains unchanged. This mode requires the validated vLLM 0.26 Ascend
dynamic-KV layout and matching model geometry and weight version. It supports one
continuation per request, without repetition/presence/frequency penalties, per-request
seeded RNG, stop strings, or prompt logprobs. Unsupported states fail closed; this is
not a general TP migration implementation.

```bash
python3 benchmark/npu_share/run_bench.py --rl math --engine fsdp \
  --batches 6 --concurrency 16 --running-rebalance kv
```

Queued-request stealing retains priority. When it selects nothing, the planner scans a
snapshot and compares bounded KV subsets against leaving requests in place. It requires
exact native computed-token counts, destination KV headroom, and predicted
completion-time improvement after transfer/planning costs and a margin. The common
transfer barrier and possible worker stalls are charged to both peers. Training
readiness is rechecked during staging. If the gate rejects all candidates, no KV is
moved; therefore an enabled run is not by itself evidence of migration or of a speedup.

Request traces include `kv_rebalance_plan`, `kv_rebalance_snapshot`,
`kv_rebalance_handoff`, and `kv_import_result`. A transport call alone does not prove
successful KV reuse: check `import_applied` at the destination. The
`kv_handoff_probe.py` tool performs a deterministic continuation comparison on actual
vLLM engines. Current correctness evidence covers the tested Qwen2.5 model; performance
and absence of regressions must be evaluated separately.

For controlled ablations, the internal environment setting
`AREAL_KV_REBALANCE_EXECUTE=0` keeps the same KV-mode memory policy, transport
initialization, snapshot collection, and forecast, but returns before reserving or
moving requests. The default is `1`. This environment setting must be passed explicitly
into the experiment container; a host-shell export alone is not forwarded by the
benchmark launcher. Record the override with the run results.

### Experimental aggressive KV migration policy

`AREAL_KV_REBALANCE_POLICY=aggressive` is an opt-in runtime experiment used with
`AREAL_RUNNING_REBALANCE=kv`. The KV-mode default is the accepted moderated
`exploratory` policy. Set this environment variable inside the training container when
using a custom experiment driver; the standard benchmark CLI does not forward arbitrary
host environment variables.

The policy optimizes question-group readiness when the visible horizon is complete,
otherwise mean request completion. It uses a common staging barrier, separate per-device
stalls, rolling measured costs, a small net-gain margin, and bounded predicted tail
regression. Version, lease, prefix, and memory checks remain required. See
[the experiment protocol](../../docs/research/npu-sharing/experiments/2026-10-19-r001-aggressive-policy.md)
for initial cost assumptions and prediction limitations.

### Development tooling: persistent vLLM paired repetitions

Implemented in `run_reuse_dev_bench.py`. This is benchmark tooling, not an NPU-sharing
research contribution.

The requested execution order is:

1. Prepare the environment and initialize reusable inference services.
1. Run fixed-card RL, repetition 1.
1. Run shared-card RL, repetition 1.
1. Run fixed-card RL, repetition 2.
1. Run shared-card RL, repetition 2.

Before each run, restore the same initial model and reset training/optimizer state,
sampling state, queues, KV contents, handoff ownership, weight versions, scheduler
calibration, and metrics. Verify equivalent warmup state. Keep model, TP layout, device
placement, input data, and tracing comparable. Changing an incompatible engine topology
requires recreation rather than unsafe reuse.

Report each fixed/shared pair separately, including training-loop time, tokens,
completed answers, migrations, and errors; provide an aggregate only in addition. Record
initialization, reset, and warmup costs separately. Two repetitions improve visibility
into variation but do not establish statistical significance.

Run development repetitions with:

```bash
python3 benchmark/npu_share/run_reuse_dev_bench.py --batches 6 --concurrency 16 \
  --running-rebalance kv --kv-policy exploratory --repetitions 2
```

This separate tool keeps topology-specific service pools, restores an immutable
initial-weight DRAM snapshot independently of current-weight offload storage, checks
weight digests and worker identity, and starts a fresh trainer per arm. Results are
explicitly development-only. Use `run_bench.py` with independent engine startup for
final confirmation; its execution flow does not reuse vLLM. See the
[accepted policy record](../../docs/research/npu-sharing/experiments/2026-10-19-r001-accepted-policy.md).
