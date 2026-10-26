[中文](M3-asymmetric-publication.zh.md) | [Overview](../core-mechanisms.md)

# M3: Asymmetric asynchronous weight publication

## Problem and mechanism

Resume ready inference replicas promptly without blocking the whole training loop on
shared-replica snapshot and restoration work.

Stable weights → independent dedicated HCCL and shared snapshot readers → per-replica
version acknowledgement → independent resume → join before weight mutation.

## Traceable implementation records

- [R003](../optimizations/R003-asymmetric-publication.md)
- [R008](../optimizations/R008-flat-weight-snapshots.md)
- [R009](../optimizations/R009-independent-snapshot-reader.md)
- [R010](../optimizations/R010-background-step-tail.md)
- [R011](../optimizations/R011-terminal-publication-elision.md)

These are components of one mechanism, not independent novelty claims. Stable R
identifiers and historical evidence remain unchanged.

## Limitations and negative results

An early R010 overlap run regressed; independent threads do not remove memory-bandwidth
contention. R011 elides publication only at actual termination when evaluation does not
need updated weights, not at every epoch boundary.

## Required isolated validation

Ablate flat snapshots, independent readers, per-replica resume and background tails
separately; measure both publication branches, training critical path and version
correctness.

Overall shared-system speedup does not establish the isolated benefit or novelty of this
mechanism. See the
[accepted-policy evidence](../experiments/2026-10-19-r001-accepted-policy.md).
