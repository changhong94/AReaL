English | [中文](README.zh.md)

# NPU sharing research ledger

This review reconstructs **24 research entries** from original session messages, code
and retained experiments. Each has English/Chinese notes covering motivation, mechanism,
contribution candidates, challenges, code, outcomes, evidence and next validation. R001
remains under ideas; the rest are under optimizations.

**Not every entry is a successful optimization**: R001 is proposed, R019 includes a
removed heuristic, R020 is a negative result, R023 is infrastructure, and R024 is
upstream context. Existing R001–R006 IDs are preserved.

Start with [evidence methodology](methodology.md),
[historical experiment catalog](experiments/history-catalog.md), and
[evolution and paper candidates](roadmap.md).

| ID   | Research entry                                                                                                           | Current conclusion                                                                                                   |
| ---- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| R001 | [Cost-aware running-request rebalancing](ideas/R001-runtime-request-rebalancing.md)                                      | Proposed; E001 supports the motivation, not a speedup.                                                               |
| R002 | [Training-priority inference on the same NPU](optimizations/R002-training-priority-concurrency.md)                       | Implemented, opt-in; functional overlap verified; isolated speedup remains workload-dependent.                       |
| R003 | [Asymmetric publication to dedicated and shared inference](optimizations/R003-asymmetric-publication.md)                 | Implemented; evolved further in R009 and R010.                                                                       |
| R004 | [Adaptive draining with a rolling completion-time model](optimizations/R004-adaptive-draining.md)                        | Implemented; three-window policy retained, sibling-length scaling removed.                                           |
| R005 | [Cost-aware rebalancing of unexecuted requests](optimizations/R005-queued-request-rebalancing.md)                        | Implemented; no eligible requests moved in the two inspected comparison runs.                                        |
| R006 | [Demand-mapped physical KV pages with slab packing](optimizations/R006-dynamic-kv-pages.md)                              | Implemented for validated Ascend layouts; memory savings verified, latency gain not established in isolation.        |
| R007 | [Single-NPU switching and borrowing idle training capacity](optimizations/R007-single-npu-and-borrowing.md)              | Implemented foundation; single-card capacity saving is not a speedup claim.                                          |
| R008 | [Flat versioned weight snapshots](optimizations/R008-flat-weight-snapshots.md)                                           | Implemented; historical publication cost reduction observed.                                                         |
| R009 | [Independent snapshot reader without HCCL queue blocking](optimizations/R009-independent-snapshot-reader.md)             | Implemented; direct publication timings improved in a short comparison.                                              |
| R010 | [Move shared publication and step-tail work off the main loop](optimizations/R010-background-step-tail.md)               | Implemented; overlap verified, historical first comparison regressed.                                                |
| R011 | [Skip unused terminal inference restoration](optimizations/R011-terminal-publication-elision.md)                         | Implemented with evaluation-aware finality.                                                                          |
| R012 | [Lease-based routing and cross-card continuation](optimizations/R012-request-leases-and-migration.md)                    | Implemented default migration path; KV is rebuilt, not transferred.                                                  |
| R013 | [Question affinity and global rollout concurrency](optimizations/R013-question-affinity-and-concurrency.md)              | Affinity implemented as an option; higher concurrency helped one workload, affinity itself did not.                  |
| R014 | [Optimizer prewarm and side-effect-free memory status](optimizations/R014-optimizer-prewarm-and-pure-status.md)          | Implemented; first-step replay improvement observed.                                                                 |
| R015 | [Address-preserving training-cache parking and prefetch](optimizations/R015-training-cache-parking-prefetch.md)          | Implemented experimental switches; mechanism verified, exact numerical equivalence not established by timing replay. |
| R016 | [Warm real Ascend request kernels before readiness](optimizations/R016-ascend-kernel-warmup.md)                          | Implemented for supported APIs; applies to fixed and shared modes.                                                   |
| R017 | [Pressure-driven partial offload of resident state](optimizations/R017-pressure-driven-partial-offload.md)               | Implemented at safe transition boundaries, not arbitrary operator-time paging.                                       |
| R018 | [Retained same-card pause with partial KV backup](optimizations/R018-retained-pause.md)                                  | Implemented experimental alternative; migration remains the requested default.                                       |
| R019 | [Tail estimation and the rejected sibling-scaling heuristic](optimizations/R019-tail-estimation-and-sibling-ablation.md) | Tail/dynamic-service refinements retained; sibling scaling removed; startup ranking remains offline research.        |
| R020 | [Prewarmed TP=2 for remaining long-tail requests](optimizations/R020-tp-tail-negative-result.md)                         | Measured inference-only alternative; dynamic TP/KV migration was not implemented.                                    |
| R021 | [Megatron integration for shared execution](optimizations/R021-megatron-backend-adaptation.md)                           | Implemented and validated on the recorded small-model configuration.                                                 |
| R022 | [Multiple inference instances and per-rank TP memory budgets](optimizations/R022-multi-npu-rank-budgets.md)              | Recent uncommitted extension; 4-card TP1 and 8-card TP2 functional coverage, no matched scaling claim.               |
| R023 | [Optional tracing and a reproducible comparison harness](optimizations/R023-observability-and-reproducibility.md)        | Implemented infrastructure; tracing itself can perturb performance.                                                  |
| R024 | [Asynchronous rollout and multi-version experiments: context](optimizations/R024-async-and-multiversion-context.md)      | Historical exploration; upstream capability and this project’s extensions must be separated.                         |

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

## Latest experiment

[E001: Megatron c32, six-batch comparison](experiments/20261015-144720-megatron-c32-b6.md)
