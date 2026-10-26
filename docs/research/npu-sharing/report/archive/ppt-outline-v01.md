# Presentation outline: NPU sharing in asynchronous RL

**Status: draft v0.1, awaiting user review; deck generation is not approved.**

## Proposed scope

- Audience: internal systems and RL researchers; approximately 20 minutes.
- Proposed 15 main slides plus optional appendix. Length, language, emphasis and
  template remain reviewable.
- Organize around M1/M2/M3; retain R001–R024 as traceable records rather than 24
  contribution slides.
- Proposed figures use English; final deck language is pending review.
- Generate the deck only after explicit outline approval, then submit the deck draft for
  review and revision.

## 01. Research goal: shorten the asynchronous RL training loop

- Main message: Define workload, scope and objective; do not claim upstream asynchronous
  RL as new work.
- Proposed visual: training and inference lanes
- Evidence/source: [record](../core-mechanisms.md)
- Review notes: pending.

## 02. Where fixed allocation leaves waiting time

- Main message: Training capacity can idle during rollout; publication and memory
  transitions can delay progress.
- Proposed visual: measured phase timeline with waits
- Evidence/source: [record](../optimizations/R007-single-npu-and-borrowing.md)
- Review notes: pending.

## 03. System design: three core mechanisms

- Main message: M1 provides safe capacity; M2 schedules work for training readiness; M3
  makes version-ready replicas available.
- Proposed visual: three-module interaction diagram
- Evidence/source: [record](../core-mechanisms.md)
- Review notes: pending.

## 04. M1: training-priority device sharing

- Main message: Borrow training capacity during rollout waits; keep the training group
  fixed and enforce transition fences.
- Proposed visual: fixed versus shared role timeline
- Evidence/source: [record](../mechanisms/M1-resource-memory-sharing.md)
- Review notes: pending.

## 05. M1: demand-driven memory and restoration

- Main message: Combine dynamic KV pages, partial offload and training-state prefetch;
  distinguish live state from inactive cache.
- Proposed visual: HBM–DRAM state diagram
- Evidence/source: [record](../mechanisms/M1-resource-memory-sharing.md)
- Review notes: pending.

## 06. M2: optimize training readiness

- Main message: Estimate data and device readiness separately; compare draining and
  migration against their maximum using completed-batch statistics.
- Proposed visual: prediction–decision–feedback
- Evidence/source: [record](../mechanisms/M2-readiness-aware-scheduling.md)
- Review notes: pending.

## 07. M2: moderated running-request KV migration

- Main message: Prioritize queued work; exploratory probes allow five requests and two
  attempts, stopping new probes after two cumulative seconds. Positive-gain plans are
  separate.
- Proposed visual: freeze–transfer–import–resume
- Evidence/source: [record](../experiments/2026-10-19-r001-accepted-policy.md)
- Review notes: pending.

## 08. M3: asymmetric asynchronous publication

- Main message: Resume each replica after its own version acknowledgement; independent
  snapshot and HCCL readers join before subsequent weight mutation.
- Proposed visual: publication branches and join points
- Evidence/source: [record](../mechanisms/M3-asymmetric-publication.md)
- Review notes: pending.

## 09. Correctness and supported scope

- Main message: Enforce unique ownership, weight/KV consistency, memory budgets and
  handoff fences; current live KV scope is two devices with TP1.
- Proposed visual: constraint table
- Evidence/source: [record](../optimizations/R001-runtime-request-rebalancing.md)
- Review notes: pending.

## 10. Evaluation: separate development and final confirmation

- Main message: Reusable vLLM is development tooling; final comparisons independently
  initialize engines. Freeze model, source, workload, tracing and placement.
- Proposed visual: two evaluation flows
- Evidence/source: [record](../methodology.md)
- Review notes: pending.

## 11. Independent-startup comparison: 14.72% lower loop time

- Main message: Fixed 179.232 s versus shared 152.847 s; 576 answers, 137098/138424
  tokens, ten confirmed imports. Report a single observation, not statistical
  significance.
- Proposed visual: phase comparison and metric table
- Evidence/source:
  [record](../experiments/evidence/r001-moderated-20261019/accepted.json)
- Review notes: pending.

## 12. Requests and devices: where does the improvement occur?

- Main message: Use request timelines, token throughput and utilization from the same
  run; explain overlap and transfer costs without mixing experiments.
- Proposed visual: same-run traces
- Evidence/source: [record](../experiments/2026-10-19-r001-accepted-policy.md)
- Review notes: pending.

## 13. What unsuccessful attempts teach us

- Main message: The timing gate fired zero times; sibling scaling was removed; tested
  TP2 tails regressed. Distinguish engineering faults from negative algorithmic
  evidence.
- Proposed visual: alternatives and evidence table
- Evidence/source: [record](../core-mechanisms.md)
- Review notes: pending.

## 14. Evidence limits and next ablations

- Main message: The 14.72% is a whole-system result, not isolated KV benefit. Match
  sharing configurations for migration ablation and extend repetitions, models and
  quality checks.
- Proposed visual: verified versus pending matrix
- Evidence/source: [record](../roadmap.md)
- Review notes: pending.

## 15. Summary: three mechanisms target the critical path

- Main message: Recap resource sharing, readiness scheduling and asynchronous
  publication; distinguish accepted policy, alternatives and next validation targets.
- Proposed visual: reuse three-mechanism overview
- Evidence/source: [record](../core-mechanisms.md)
- Review notes: pending.

## Optional appendix

- A1: R001–R024 mapping to M1/M2/M3.
- A2: Actual configuration, source commits, model and image identity; extract rather
  than guess.
- A3: Ownership, KV handoff failures and version correctness.
- A4: Detailed negative results and rejected alternatives.
- A5: Tool operation and reused-vLLM comparability limits.

## Review checklist

- [ ] Audience, duration, deck language and working title.
- [ ] Order and emphasis of M1/M2/M3.
- [ ] Results, negative evidence and limitations.
- [ ] Slide edits, visuals and template preference.
- [ ] Explicit user approval to generate the presentation.
