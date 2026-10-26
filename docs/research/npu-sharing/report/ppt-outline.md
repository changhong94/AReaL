# Presentation outline: NPU sharing in asynchronous RL

**v0.2 review draft, 2026-10-21. The user requested outline revision and deck
generation. The deck awaits review.**

English deck, 18 main slides including cover, plus two appendix slides. Internal systems
and RL audience, approximately 20–25 minutes.

[中文](ppt-outline.zh.md)

## Changes since v0.1

- Retain M1/M2/M3 as the three mechanisms, with R001–R028 as traceable records.
- Add current R025 admission scaling and R027 return decisions and resident-cache
  budgets.
- Update full inference graphs, equal-TP, multiple-replica and paused-KV validation
  scope.
- Separate the historical 14.72% result, R025 development evidence (15.40%), and formal
  graph pairs (9.01%/8.96%).
- Record the Megatron/R027 repair: 30 CPU tests passed, NPU confirmation pending.
- Preserve negative evidence and identify R026/R028 as proposals.

## 01. NPU Sharing for Asynchronous RL

Three mechanisms coordinate resource sharing, request admission and version publication.
Internal research review, draft v0.2.

Sources: [core-mechanisms.md](../core-mechanisms.md)

## 02. The objective is earlier training progress

Optimize whole-loop time while preserving training semantics. More active devices alone
do not establish a benefit.

Sources: [methodology.md](../methodology.md),
[R007-single-npu-and-borrowing.md](../optimizations/R007-single-npu-and-borrowing.md)

## 03. Three core mechanisms

M1 provides safe capacity. M2 assigns and admits useful work. M3 exposes replicas only
after their own publication acknowledgement. R025 and R027 extend this architecture.

Sources: [core-mechanisms.md](../core-mechanisms.md),
[R025-adaptive-inference-concurrency.md](../optimizations/R025-adaptive-inference-concurrency.md),
[R027-profit-aware-inference-return.md](../optimizations/R027-profit-aware-inference-return.md)

## 04. M1: borrow training capacity safely

Device roles change while the training process group and optimizer ownership remain
fixed.

Sources:
[R007-single-npu-and-borrowing.md](../optimizations/R007-single-npu-and-borrowing.md)

## 05. M1: reclaim memory only when needed

Dynamic KV mapping, partial offload and prefetch retain useful state when physical
memory allows. Budget checks remain mandatory.

Sources:
[R017-pressure-driven-partial-offload.md](../optimizations/R017-pressure-driven-partial-offload.md),
[R006-dynamic-kv-pages.md](../optimizations/R006-dynamic-kv-pages.md),
[R015-training-cache-parking-prefetch.md](../optimizations/R015-training-cache-parking-prefetch.md)

## 06. M2: predict data and device readiness

Use statistics from up to three completed batches. The next training step needs both
complete answer groups and the training device.

Sources: [R004-adaptive-draining.md](../optimizations/R004-adaptive-draining.md),
[R019-tail-estimation-and-sibling-ablation.md](../optimizations/R019-tail-estimation-and-sibling-ablation.md)

## 07. M2: move running work with bounded KV transfer

R001 first considers queued work, then compatible running requests. Accepted exploration
permits five requests per probe, two attempts and a two-second cumulative transaction
admission budget. Positive-gain plans are separate.

Sources:
[R001-runtime-request-rebalancing.md](../optimizations/R001-runtime-request-rebalancing.md),
[2026-10-19-r001-accepted-policy.md](../experiments/2026-10-19-r001-accepted-policy.md)

## 08. R025: admission follows available capacity

Current target ceil(C × R / F), with equivalent compatible replicas. C is baseline
question concurrency, R ready replicas, F dedicated replicas. TP ranks are not replicas.

Sources:
[R025-adaptive-inference-concurrency.md](../optimizations/R025-adaptive-inference-concurrency.md)

## 09. R027: return to inference only when useful

Decide at training completion using current batch readiness and recent measured
transition costs. Missing evidence triggers calibration. Formal results primarily
validate the next-batch-ready shortcut.

Sources:
[R027-profit-aware-inference-return.md](../optimizations/R027-profit-aware-inference-return.md),
[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md)

## 10. M3: publish to each replica independently

Independent HCCL and snapshot readers remove unnecessary cross-branch waiting. Each
branch must acknowledge before its admission opens. Actor mutation waits for active
readers.

Sources:
[R009-independent-snapshot-reader.md](../optimizations/R009-independent-snapshot-reader.md),
[R010-background-step-tail.md](../optimizations/R010-background-step-tail.md)

## 11. Correctness defines the migration boundary

Ownership, version consistency and rank-local budgets take priority over load balancing.
Paused same-version TP1 and TP2 probes preserve 256 deterministic tokens and exact
logged probabilities.

Sources:
[R027-profit-aware-inference-return.md](../optimizations/R027-profit-aware-inference-return.md),
[R001-runtime-request-rebalancing.md](../optimizations/R001-runtime-request-rebalancing.md)

## 12. Evaluation separates exploration from confirmation

Reused engines accelerate exploration but can change runtime state. Formal comparisons
initialize each arm independently. Source, workload and observation switches must match.

Sources: [methodology.md](../methodology.md),
[R023-observability-and-reproducibility.md](../optimizations/R023-observability-and-reproducibility.md)

## 13. Earlier shared-system result: 14.72% less loop time

Independent-start comparison on 2026-10-19: fixed 179.232 s, shared 152.847 s, 576
answers per arm, 137098/138424 tokens, ten confirmed imports. This is a historical
configuration, not the current graph result or an isolated R001 ablation.

Sources:
[2026-10-19-r001-accepted-policy.md](../experiments/2026-10-19-r001-accepted-policy.md),
[accepted.json](../experiments/evidence/r001-moderated-20261019/accepted.json)

## 14. R025 development result: higher useful concurrency

Static-16 shared loops averaged 181.08s, capacity policy 153.20s, a 15.40% reduction.
Two reused-engine trials per mode. Tokens fell 1.72%. Static32 and independent-start
controls remain pending.

Sources:
[2026-10-20-r025-capacity-admission.md](../experiments/2026-10-20-r025-capacity-admission.md)

## 15. Graph-mode confirmation: about 9% less loop time

Two independent-start FSDP comparisons with full inference graphs and eager training. 96
questions, six answers, batch size16, concurrency16, six batches, offpolicy2.
Savings9.01% and8.96%, aggregate8.99%. These validate the combined shared configuration.

Sources:
[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md),
[2026-10-21-r027-formal.json](../experiments/2026-10-21-r027-formal.json),
[2026-10-21-r027-formal-repeat.json](../experiments/2026-10-21-r027-formal-repeat.json)

## 16. Support and validation have different scopes

Graph-mode support is validated for the recorded two-NPU FSDP workload. Equal-TP
multi-peer transport probes establish functional correctness, not scaling benefit.
Today’s Megatron R027 publication fix passed30 CPU tests and awaits NPU confirmation.

Sources:
[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md),
[2026-10-19-r001-tp2-megatron.md](../experiments/2026-10-19-r001-tp2-megatron.md),
[R001-runtime-request-rebalancing.md](../optimizations/R001-runtime-request-rebalancing.md)

## 17. Negative results shape the design

Retain removed sibling scaling, non-triggering gates, memory overhead and
context-dependent TP results. Graph Qwen2-7B screening improved TP2
completion15.21–28.57%, but cross-TP live migration is unimplemented.

Sources:
[R019-tail-estimation-and-sibling-ablation.md](../optimizations/R019-tail-estimation-and-sibling-ablation.md),
[R020-tp-tail-negative-result.md](../optimizations/R020-tp-tail-negative-result.md),
[2026-10-20-r026-qwen2-7b-graph-tp-screen.md](../experiments/2026-10-20-r026-qwen2-7b-graph-tp-screen.md),
[2026-10-21-r027-resident-training.md](../experiments/2026-10-21-r027-resident-training.md)

## 18. Next evidence for a paper

The strongest current claim is whole-system improvement on recorded workloads. Next work
isolates the three mechanisms, tests stronger baselines and quality, and extends support
carefully.

Sources: [roadmap.md](../roadmap.md), [methodology.md](../methodology.md),
[R026-tail-acceleration-via-tp-migration.md](../optimizations/R026-tail-acceleration-via-tp-migration.md),
[R028-multinode-training-inference-sharing.md](../optimizations/R028-multinode-training-inference-sharing.md)

## 19. Appendix: current benchmark defaults

Current defaults differ from the16×6 evidence workload. Coverage varies batch
sizes16/32/64 and answer counts4/8/16.

Sources: [README.md](../../../../benchmark/npu_share/README.md),
[COVERAGE_PLAN.md](../../../../benchmark/npu_share/COVERAGE_PLAN.md)

## 20. Appendix: evidence and implementation record

Research sources live under docs/research/npu-sharing. The deck records updated status
without overwriting historical findings.

Sources: [README.md](../report/README.md)

## Review checklist

- [ ] Narrative and technical depth.
- [ ] Explanation of the three mechanisms.
- [ ] Evidence limits, negative results and next research.
- [ ] Figures, wording and layout.
- [ ] Mark final only after user approval.
