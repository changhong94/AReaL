English | [中文](R026-tail-acceleration-via-tp-migration.zh.md)

# R026: Accelerating live long tails by migrating to a larger-TP engine

Status: Proposed; first inference-only screening completed with a negative result;
cross-TP migration not implemented. Recorded: 2026-10-20 (America/Vancouver), from the
user's proposal in this session. Primary mechanism: M2 scheduling; M1 supplies
device/memory feasibility and M3 supplies weight-version readiness. This is not a fourth
core mechanism.

## Problem and proposal

A few long autoregressive requests can delay an entire rollout batch after other
requests have finished. Additional independent replicas cannot shorten a single
request's sequential decode chain. Prepare and warm a larger-TP inference engine before
it is needed, then migrate an already-running tail request, including its KV cache and
continuation state, to that engine to finish sooner.

The candidate model must benefit from larger TP in low-concurrency decode, not merely
achieve higher aggregate throughput at large batch sizes. Screen Qwen2.5-7B-Instruct
first; retain Qwen2.5-1.5B-Instruct as the R020 negative control. The 7B recommendation
is a hypothesis, not established hardware evidence.

## Decision rule

Move only when expected remaining time saved exceeds all incremental costs:

`gain = remaining_time_source - (quiesce + KV_reshard_transfer + import_resume + remaining_time_target)`

Also account for target queueing, uncertainty and impact on other requests or training.
A prewarmed engine removes startup from the immediate migration path, but its reserved
NPU time, HBM, weights and opportunity cost must still be reported. Use remaining-length
and measured service-time distributions, not an oracle that uses the eventual output
length. Re-evaluate at safe decode boundaries.

## Correctness and engineering challenges

- Source and target must hold the exact same model, weight version, tokenizer,
  positional encoding and compatible KV precision/layout. Pin the request version.
- TP=1 to TP=2 changes KV head ownership and possibly replication (GQA/MQA). Convert
  logical layer/head/token positions into destination shards. Copying a contiguous byte
  range is not a general cross-TP migration protocol.
- Quiesce at an acknowledged token boundary; transfer committed token IDs, sequence
  length, block mapping, sampling/RNG state, stop criteria and required logprob state.
- Reserve destination pages on every TP rank, transfer with a matching communication
  protocol, and acknowledge complete import before atomically transferring the lease.
  Retain the source state until commit; failures must permit safe source resumption
  without duplicate/missing tokens or double scheduling.
- Different TP reductions can change floating-point results. Validate logprob error and
  greedy continuations separately from sampled trajectory equality.
- Spare cards and a prewarmed TP engine are not free. Do not steal training capacity or
  overwrite active inference weights without the existing budget/version gates.

## Relation to existing work and code locations

R020 found prewarmed TP2 slower for the prior 1.5B tail workload; it did not implement
KV migration. Preserve that result. R001 currently covers compatible same-TP migration,
not this cross-TP protocol. R019 supplies tail forecasts, R025 controls admission, and
R022 provides per-rank feasibility. Candidate extension points:
`areal/infra/controller/rollout_controller.py`, `areal/infra/kv_rebalance.py`,
`areal/infra/request_router.py`, and the existing vLLM KV handoff implementation.
Inspect actual layout and protocol interfaces before implementing cross-TP support.

## Staged validation

1. Prepare a pinned model snapshot; record software/image identity and hardware.
1. On prewarmed TP1 and TP2, replay identical tokenized inputs with fixed output
   lengths. Test one tail and small groups, plus concurrency 16/32 for context. Separate
   TTFT, decode TPOT, completion time, output count, startup and NPU-seconds.
1. Only if remaining decode is faster, implement cross-TP import correctness tests.
   Measure transfer size/time at multiple KV lengths and per-rank memory pressure.
1. Compare stay-on-source, restart/recompute migration and KV-preserving migration.
   Compare against an already-TP2 baseline and report extra reserved resource cost.
1. Integrate causal scheduling into RL; measure batch-ready and training-loop time,
   quality, version correctness, false-positive migrations and regressions. Use repeated
   exploratory runs, then independently initialized formal confirmation.

No benefit is claimed yet. If TP2 decode is slower before transfer, this model and
request shape fail the prerequisite; retain the negative evidence rather than building a
migration path on an assumed speedup.

## Model preparation update (2026-10-20)

Qwen2.5-7B-Instruct was the first candidate, but the local directory had no weights and
external download was slow. The first hardware screening instead uses the complete local
Qwen2-7B-Instruct snapshot after official LFS SHA-256 verification. These are different
models; do not label Qwen2 measurements as Qwen2.5 results. The default RL benchmark
model remains Qwen2.5-1.5B-Instruct.

## First screening result

TP2 was slower for every measured Qwen2-7B-Instruct workload. Cross-TP KV migration
remains unimplemented. [Evidence](../experiments/2026-10-20-r026-qwen2-7b-tp-screen.md).
