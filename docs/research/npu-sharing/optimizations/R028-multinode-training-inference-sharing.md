English | [中文](R028-multinode-training-inference-sharing.zh.md)

# R028: Multi-node RL training–inference sharing and scheduling

Status: Proposed; the shared multi-node execution path is not implemented or validated.
Recorded: 2026-10-20 (America/Vancouver), at the user's request. Scope: extend M1, M2
and M3 across nodes; this is not a fourth core mechanism.

## Motivation

Single-node results do not establish cluster support. A cluster may have idle training
devices while another node is serving a long rollout backlog. Borrowing devices,
expanding inference admission and rebalancing work across nodes could reduce batch-ready
time, but network transfer, collective synchronization and stragglers can erase those
gains. The target is end-to-end RL loop time under correctness and training-priority
constraints, not utilization alone.

## Current boundary, verified in code

`PPOTrainer._validate_npu_share()` requires a single controller and local scheduler,
pure-DP FSDP or Megatron training (training TP=1 and PP=1), and colocated forked vLLM
workers. Shared vLLM uses `uni` for TP1 and local `mp` for TP>1. Existing single-host
8-NPU/TP2 evidence refers to inference TP, not cross-node training TP.
`areal/infra/scheduler/ray.py` has multi-node engine launch paths, but they do not
establish support for the added sharing lifecycle. The benchmark driver currently uses
one host and local Docker/device mappings.

Treat these as distinct cases:

1. Multiple independent inference instances placed on different nodes.
1. One inference instance with TP/PP spanning nodes.
1. A distributed training group spanning nodes, with selected training devices borrowed
   for inference between or during training phases.
1. Same-version running-request KV migration between engines on different nodes.

Each needs separate validation. Do not simply remove the local-scheduler guard.

## Proposed mechanism extensions

### M1: Node-local memory control, globally coordinated readiness

Keep HBM/DRAM ownership, reusable-cache evidence, pressure reclamation and graph-address
stability local to each device. Advertise per-rank usable capacity and transition state
to a coordinator. A training collective may begin only after every required rank has
acknowledged restored weights, optimizer state, workspace and applicable inference
fencing. Device identity must include node identity. Missing or timed-out
acknowledgements must fail closed; no rank may enter a collective while another remains
in inference. Do not conflate borrowing a fixed training group with resizing that group.

### M2: Hierarchical placement and migration cost

Prefer node-local placement and queued-request relocation first. Extend R025's
ready-capacity signal across nodes; heterogeneous engines need measured effective
capacity rather than equal replica counting. Extend R001 only after validating
cross-node HCCL rendezvous and payload correctness. Estimate transfer from actual KV
bytes, measured path bandwidth, latency, queueing and import time, including contention
with training communication. Keep node-local and remote candidates separate. A migration
requires compatible model/version/layout, destination capacity on all TP ranks, a single
request lease and a complete import acknowledgement. Preserve the source until the
transfer commits; retain failures as evidence. R027 should include remote publication
and future restoration costs in its critical-path decision.

### M3: Per-replica version readiness across nodes

Remove assumptions that a local snapshot path is visible to all receivers. Choose an
explicit shared-filesystem or network-transfer protocol, with immutable version
identity, integrity checks and snapshot ownership until readers finish. Resume each
ready replica after its own load acknowledgement; do not overwrite weights still
required by live requests. Verify graph replay after updates and memory remapping on
every participating rank. Coordinate source buffer lifetime and failure handling across
nodes without blocking unrelated ready inference replicas unnecessarily.

## Candidate code locations

- `areal/trainer/rl_trainer.py`: supported-layout validation and phase transitions.
- `areal/trainer/colocation_memory.py`, engine memory backends: per-rank capacity and
  safe restoration.
- `areal/infra/scheduler/ray.py`, train/rollout controllers: placement, node/device
  identity and readiness acknowledgements.
- `areal/infra/adaptive_concurrency.py`, request router, rolling predictors and
  `kv_rebalance.py`: capacity, placement and cost-aware migration.
- `areal/engine/vllm_ext/kv_handoff.py` and weight-publication adapters: transport,
  ownership and version correctness.
- Benchmark tooling: multi-host inventory, synchronized clocks, per-node traces and
  failure cleanup. Tooling is separate from the research contribution.

These are investigation points, not a promise that edits to these files alone are
sufficient.

## Staged implementation and validation

1. **Two-node fixed baseline.** Establish cross-node pure-DP training and independent
   node-local inference engines. Verify network/collectives, identical
   model/data/configuration, synchronized clocks and trace completeness.
1. **Sharing with node-local migration only.** Retain each inference TP group within one
   node; distribute training ranks across nodes. Validate all-rank readiness and local
   memory decisions, including asymmetric pressure and graph mode.
1. **Cross-node queued and running-request relocation.** Begin with same-TP engines and
   same-version requests. Sweep KV length, network load and bandwidth; inject timeout,
   receiver failure and partial acknowledgement faults.
1. **Cross-node inference TP.** Separately validate one inference instance spanning
   nodes, grouped pause/resume and graph collectives. This does not automatically
   provide R026's live migration between different TP widths.
1. **Broader training layouts.** Training TP/PP extensions are later work, beyond the
   initial pure-DP scope.

For each phase compare fixed allocation, sharing without remote migration, and sharing
with remote migration under identical resources. Report loop and batch-ready times,
reward/evaluation quality, token counts, off-policy age, NPU-seconds, per-node
utilization/HBM, network bytes, collective wait, publication delay and successful versus
attempted migrations. Use repeated development runs and independent-start formal
confirmation. Include negative cases where communication overhead exceeds benefit.
Reusing engines is an experiment tool, not a research result.

## Hypotheses, risks and evidence status

Potential benefits: recover capacity across uneven node loads, shorten rollout tails and
overlap publication without global stalls. Risks: training-rank stragglers, network
contention, stale capacity reports, duplicate execution after failure and graph-address
violations. No multi-node speedup, correctness or fault-tolerance claim is established.
Preserve failed or non-beneficial trials alongside positive results.
