English | [中文](roadmap.zh.md)

# Evolution and candidate paper structure

The current paper organization is [three core mechanisms](core-mechanisms.md): M1
resource/memory sharing, M2 readiness-aware scheduling, and M3 asymmetric publication.
Historical stages below are not separate contribution claims; retain negative evidence
and tooling as supporting material.

## Recovered evolution

| Phase                             | Research IDs                 | Decision or lesson                                                                                                             |
| --------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Async/multi-version feasibility   | R024                         | Upstream async is the baseline; coexistence alone does not justify an extra device.                                            |
| Single-card to borrowed-card pool | R007, R012                   | Safe state transitions and request ownership are prerequisites; early sharing was slower.                                      |
| Publication critical path         | R008, R003, R009, R010, R011 | Flat snapshots, per-replica readiness, independent readers, background tail and terminal elision remove distinct dependencies. |
| Allocation and warmup             | R014, R015, R016             | Isolated replay identified first-use optimizer and backward allocation; kernel JIT explained apparent low request concurrency. |
| Residency and concurrency         | R006, R017, R018, R002       | Dynamic physical KV enables smaller residency; partial offload and concurrent admission trade memory for time.                 |
| Batch-critical scheduling         | R013, R004, R019             | Complete questions matter; predict both data and training-card readiness. More history or sibling information can regress.     |
| Recover idle shared capacity      | R005, R020, R001             | Queued stealing found no candidates; TP2 tail replay was slower; running-request rebalancing is still a hypothesis.            |
| Generalization and evidence       | R021, R022, R023             | Megatron, multi-NPU topology, per-rank budgets and reproducible instrumentation expand applicability, not automatically speed. |

## Paper structure: three core mechanisms

Use [the three-mechanism overview](core-mechanisms.md): M1 training-priority resource
and memory sharing, M2 request scheduling for training readiness, and M3 asymmetric
asynchronous weight publication. Treat warmup, backend extensions and tools as
supporting work. Preserve failed and non-beneficial alternatives as limitations,
ablations and discussion evidence.

These are three candidate mechanisms, not three proven novel contributions with isolated
speedups. Record related work and FineServe/CaMem/Ascend attribution.

## Missing work before strong claims

- A systematic related-work matrix with actual source citations.
- Repeated full-loop experiments, equal-output replay, broader models/lengths and
  hardware pressure; separate memory capacity benefit from latency benefit.
- Same-version handoff design and fault tests for R001 before live migration
  experiments.
- Quality/off-policy evaluation, recovery semantics and longer-run stability.
- Multi-NPU matched scaling, TP4 validation and asymmetric-pressure RL (not only a
  native test).
- Exact attribution for historical runs without complete source/image provenance.

No new NPU experiment or production scheduler change was made for this review.

## Proposed multi-node extension (R028)

[Research plan](optimizations/R028-multinode-training-inference-sharing.md): two-node
fixed baseline, node-local sharing with distributed training, cross-node request
migration, then cross-node inference TP. Current npu_share remains local-only; no
multi-node benefit is claimed.

## Reward scheduling (R029)

[Question-completion-first scoring](optimizations/R029-training-ready-reward-scheduling.md):
optional prototype; replay identical generated answers first, then repeat full-loop
comparisons. Preserve the negative decoupled-loop result and external sandbox contention
caveat.
