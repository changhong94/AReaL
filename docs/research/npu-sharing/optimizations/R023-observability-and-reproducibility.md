English | [中文](R023-observability-and-reproducibility.zh.md)

# R023: Optional tracing and a reproducible comparison harness

Status: Implemented infrastructure; tracing itself can perturb performance.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Mechanism diagram

![R023: Match conditions and trace causality before claiming a speedup](R023.png)

Tracing can perturb performance. Fixed placement uses the same instrumented fork.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R023.svg) | [PNG](R023.png)

## Problem and mechanism

Aggregate utilization hides routing, pauses and compile stalls. Record
request/task/attempt IDs, actual consumed batch, engine, weight version,
queue/pause/resume/device windows, TTFT/TPOT and completion. Keep synchronized device
tracing off by default; CPU progress telemetry is separate. The runner freezes one
source snapshot, captures settings/data/image hashes, runs matched modes and produces
phase, request, utilization and token charts outside the repository.

## Fairness corrections and difficulty

Fixed mode means fixed placement in the same instrumented fork, not pristine upstream.
Match global question concurrency and sandbox reward concurrency: the first code
comparison accidentally allowed 4 versus 8 global sandbox operations and was
invalidated; the rerun used global 4 in both arms. Bound exact submitted workloads so
unused prefetch does not silently change the comparison. Never invent NPU times when
tracing is off.

## Evidence and limitations

Corrected code comparison: fixed 218.93 s / 71823 tokens, shared 211.09 s / 69395
tokens, pass counts 26/288 and 31/288. Those are rollout rewards, not an independent
before/after quality evaluation. Two-second utilization can miss peaks; HBM includes
cached/reserved bytes; token bars use single-token decode ends. Overlapping windows are
not exclusive per-request compute. Same-seed live runs may differ in outputs, batches
and gradients.

## Research value and next validation

This is the evidence infrastructure for causal claims, not a throughput optimization by
itself. Include uninstrumented repetitions, source/patch hashes, off-policy settings,
actual generated lengths, evaluation and uncertainty. Use the evidence catalog to avoid
comparing incompatible historical experiments.

## Code locations

- [areal/utils/request_trace.py](../../../../areal/utils/request_trace.py)

- [areal/engine/vllm_ext/areal_vllm_server.py](../../../../areal/engine/vllm_ext/areal_vllm_server.py)

- `benchmark/npu_share/run_bench.py`

- `benchmark/npu_share/render.py`

## Experiment evidence and conversation provenance

- `fair-overlap-trace01`: Logs/scripts are in the persistent historical archive.

- [code-three-batches02/comparison.json](../experiments/evidence/history/code-three-batches02/comparison.json)

- [queue-rebalance-comparison01/results-summary.json](../experiments/evidence/history/queue-rebalance-comparison01/results-summary.json)

Conversation excerpt IDs: C095, C121.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.

Uncommitted code paths above refer to the reviewed working tree and are not included in
this documentation commit; consult the corresponding experiment source snapshots.
