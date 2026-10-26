English | [中文](README.zh.md)

# NPU sharing research ledger

**Reports and papers: [report](report/README.md) ·
[Presentation outline (draft)](report/ppt-outline.md)**

**Start here: [Three core mechanisms](core-mechanisms.md)**

1. **M1: Training-priority resource and memory sharing** — safe borrowing, demand-driven
   memory and training-state restoration.
1. **M2: Request scheduling for training readiness** — residual prediction, draining,
   rebalancing and KV migration.
1. **M3: Asymmetric asynchronous weight publication** — per-replica readiness,
   independent publication and background restoration.

R001–R025 remain traceable records, not 25 independent contributions. Preserve failures,
no-benefit cases, mechanisms that did not fire, alternatives and engineering faults with
their tested scope. R023 is tooling; R024 is upstream context; backend/topology
adaptation supports applicability.

R001 is implemented. The accepted shared configuration has demonstrated system-level
benefit (14.72% lower loop time in the recent independent-startup comparison); isolated
KV-migration benefit still needs ablation. See the
[accepted policy and evidence](experiments/2026-10-19-r001-accepted-policy.md).

[Diagrams](diagrams.md) · [Methodology](methodology.md) ·
[Experiment history](experiments/history-catalog.md) · [Roadmap](roadmap.md)

## Historical research-entry index

| ID   | Research entry                                                                                                           | Current conclusion                                                                                                                                                 |
| ---- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R001 | [Cost-aware running-request rebalancing](optimizations/R001-runtime-request-rebalancing.md)                              | Implemented; shared-system benefit validated; isolated migration contribution pending ablation.                                                                    |
| R002 | [Training-priority inference on the same NPU](optimizations/R002-training-priority-concurrency.md)                       | Implemented, opt-in; functional overlap verified; isolated speedup remains workload-dependent.                                                                     |
| R003 | [Asymmetric publication to dedicated and shared inference](optimizations/R003-asymmetric-publication.md)                 | Implemented; evolved further in R009 and R010.                                                                                                                     |
| R004 | [Adaptive draining with a rolling completion-time model](optimizations/R004-adaptive-draining.md)                        | Implemented; three-window policy retained, sibling-length scaling removed.                                                                                         |
| R005 | [Cost-aware rebalancing of unexecuted requests](optimizations/R005-queued-request-rebalancing.md)                        | Implemented; no eligible requests moved in the two inspected comparison runs.                                                                                      |
| R006 | [Demand-mapped physical KV pages with slab packing](optimizations/R006-dynamic-kv-pages.md)                              | Implemented for validated Ascend layouts; memory savings verified, latency gain not established in isolation.                                                      |
| R007 | [Single-NPU switching and borrowing idle training capacity](optimizations/R007-single-npu-and-borrowing.md)              | Implemented foundation; single-card capacity saving is not a speedup claim.                                                                                        |
| R008 | [Flat versioned weight snapshots](optimizations/R008-flat-weight-snapshots.md)                                           | Implemented; historical publication cost reduction observed.                                                                                                       |
| R009 | [Independent snapshot reader without HCCL queue blocking](optimizations/R009-independent-snapshot-reader.md)             | Implemented; direct publication timings improved in a short comparison.                                                                                            |
| R010 | [Move shared publication and step-tail work off the main loop](optimizations/R010-background-step-tail.md)               | Implemented; overlap verified, historical first comparison regressed.                                                                                              |
| R011 | [Skip unused terminal inference restoration](optimizations/R011-terminal-publication-elision.md)                         | Implemented with evaluation-aware finality.                                                                                                                        |
| R012 | [Lease-based routing and cross-card continuation](optimizations/R012-request-leases-and-migration.md)                    | Implemented default migration path; KV is rebuilt, not transferred.                                                                                                |
| R013 | [Question affinity and global rollout concurrency](optimizations/R013-question-affinity-and-concurrency.md)              | Affinity implemented as an option; higher concurrency helped one workload, affinity itself did not.                                                                |
| R014 | [Optimizer prewarm and side-effect-free memory status](optimizations/R014-optimizer-prewarm-and-pure-status.md)          | Implemented; first-step replay improvement observed.                                                                                                               |
| R015 | [Address-preserving training-cache parking and prefetch](optimizations/R015-training-cache-parking-prefetch.md)          | Implemented experimental switches; mechanism verified, exact numerical equivalence not established by timing replay.                                               |
| R016 | [Warm real Ascend request kernels before readiness](optimizations/R016-ascend-kernel-warmup.md)                          | Implemented for supported APIs; applies to fixed and shared modes.                                                                                                 |
| R017 | [Pressure-driven partial offload of resident state](optimizations/R017-pressure-driven-partial-offload.md)               | Implemented at safe transition boundaries, not arbitrary operator-time paging.                                                                                     |
| R018 | [Retained same-card pause with partial KV backup](optimizations/R018-retained-pause.md)                                  | Implemented experimental alternative; migration remains the requested default.                                                                                     |
| R019 | [Tail estimation and the rejected sibling-scaling heuristic](optimizations/R019-tail-estimation-and-sibling-ablation.md) | Tail/dynamic-service refinements retained; sibling scaling removed; startup ranking remains offline research.                                                      |
| R020 | [Prewarmed TP=2 for remaining long-tail requests](optimizations/R020-tp-tail-negative-result.md)                         | Measured inference-only alternative; dynamic TP/KV migration was not implemented.                                                                                  |
| R021 | [Megatron integration for shared execution](optimizations/R021-megatron-backend-adaptation.md)                           | Implemented and validated on the recorded small-model configuration.                                                                                               |
| R022 | [Multiple inference instances and per-rank TP memory budgets](optimizations/R022-multi-npu-rank-budgets.md)              | Recent uncommitted extension; 4-card TP1 and 8-card TP2 functional coverage, no matched scaling claim.                                                             |
| R023 | [Optional tracing and a reproducible comparison harness](optimizations/R023-observability-and-reproducibility.md)        | Implemented infrastructure; tracing itself can perturb performance.                                                                                                |
| R024 | [Asynchronous rollout and multi-version experiments: context](optimizations/R024-async-and-multiversion-context.md)      | Historical exploration; upstream capability and this project’s extensions must be separated.                                                                       |
| R025 | [Capacity-aware adaptive inference concurrency](optimizations/R025-adaptive-inference-concurrency.md)                    | Capacity policy implemented; positive reused-engine evidence, formal confirmation pending.                                                                         |
| R026 | [Live long-tail acceleration through larger TP](optimizations/R026-tail-acceleration-via-tp-migration.md)                | Proposed; Qwen2-7B eager screening found TP2 slower; cross-TP KV migration not implemented.                                                                        |
| R027 | [Profit-aware inference return](optimizations/R027-profit-aware-inference-return.md)                                     | Opt-in prototype; development validation in progress; no established benefit.                                                                                      |
| R028 | [Multi-node training–inference sharing](optimizations/R028-multinode-training-inference-sharing.md)                      | Proposed; extends M1/M2/M3 across nodes; not implemented or validated.                                                                                             |
| R029 | [Reward scheduling for training-batch readiness](optimizations/R029-training-ready-reward-scheduling.md)                 | Work-conserving question priority; >99% slot occupancy and earlier first-batch training, but mixed whole-loop results.                                             |
| R030 | [Idle inference devices join training](optimizations/R030-idle-inference-training-expansion.md)                          | Full state validation passed; latest switch 15.31 s, development RL 2.31% faster; tokens differ, initialization excluded. Opt-in; see repeated results and limits. |

## Recording rules

1. Assign a stable research ID and record the hypothesis before changing code.

1. Link implementation commits and experiment IDs. A dirty run requires a source
   snapshot and hashes, not only its base commit. Never replace old evidence.

1. Record configurations, dataset/model identity, container image, logging settings,
   allocation, seeds, failures and negative results. Preserve actual generated answers
   when claiming identical training inputs.

1. Separate observations, explanations, hypotheses and verified causal effects.

1. Compare isolated ablations; repeat trials and report variability. Record quality and
   policy-version correctness alongside performance.

1. Update results after each experiment. Do not label a proposal as an improvement
   before it has passed correctness and performance validation.

1. Review prior work before claiming publication novelty. Keep raw evidence outside the
   repository on persistent storage; commit small summaries and manifests.

1. Maintain an English file and a matching `.zh.md` translation for every research
   document. Link both versions and update them together. Code comments remain English.

## Evidence retention

E001 is copied to `/root/areal-research-artifacts/` on host `7.150.15.118`. This is
persistent local storage, not an independent off-host backup. See its experiment record
for the exact archive path and integrity manifest. Future benchmark runs can use
`--workspace-root /root/areal-research-artifacts/runs`. The benchmark default has not
been changed.

## Historical experiment entry

[E001: Megatron c32, six-batch comparison](experiments/20261015-144720-megatron-c32-b6.md)
