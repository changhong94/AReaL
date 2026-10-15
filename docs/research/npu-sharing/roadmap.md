English | [中文](roadmap.zh.md)

# Evolution and candidate paper structure

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

## Possible paper contributions (novelty not yet established)

1. **A state and version protocol for training-priority NPU sharing.** Combine
   conditional residency, per-rank budgets and explicit request/weight fences.
1. **An independently published inference pool.** Release dedicated inference early
   while snapshot/recovery runs in the background; quantify the residual critical path.
1. **Batch-critical cost-based scheduling.** Optimize `max(data_ready, card_ready)` with
   causally frozen online estimates, then investigate running-request rebalancing.
1. **Evidence and counterexamples.** Controlled allocation diagnosis, negative TP2 and
   sibling results, and disclosure of tracing/workload differences support credibility.

These are candidate groupings, not four proven novel contributions. Slabs, virtual
memory, asynchronous RL, prefetch, survival estimation and warmup have prior art. Record
FineServe attribution and CaMem/Ascend runtime origins before drafting novelty.

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
