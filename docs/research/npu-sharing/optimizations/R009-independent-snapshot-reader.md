English | [中文](R009-independent-snapshot-reader.zh.md)

# R009: Independent snapshot reader without HCCL queue blocking

Status: Implemented; direct publication timings improved in a short comparison.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

A background consumer is insufficient if the HCCL producer blocks while submitting
tensors into its bounded snapshot queue. Give snapshot export an independent reader of
stable actor storage. HCCL only waits for its own transport/load ACK; the snapshot
thread independently converts and copies tensors on a dedicated stream. Fence the
completed optimizer update before either reader; join before next mutation or offload.

## Contribution candidate and difficulty

Remove dependency rather than just add threads. Avoid a second unsynchronized FSDP
collective traversal and bound extra converted storage to one tensor. Independent
streams can still contend for bandwidth, so overlap is not free. Multi-rank storage
requires the later explicit gather protocol; the original reader assumed a single actor.

## Measured outcome

At concurrency 16, three batches and synchronized NPU tracing off: dedicated publication
for batches 1/2 fell 4.09/4.02→2.07/1.74 s; sum 8.10→3.81 s (53% less). Loop
104.07→101.26 s (2.7% less), outputs 46650/46616 tokens. Save time also changed by about
1.21 s and question composition differed, so only publication has the clearest direct
evidence; the old fixed baseline was not rerun.

## Next validation

Retain queue submit waits, stream waits, HCCL ACK and snapshot-ready markers and repeat
with identical tensors. Measure transport contention under larger models.

## Code locations

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/engine/weight_snapshot.py](../../../../areal/engine/weight_snapshot.py)

- [tests/test_fused_snapshot_publication.py](../../../../tests/test_fused_snapshot_publication.py)

## Experiment evidence and conversation provenance

- `independent-reader01`: Logs/scripts are in the persistent historical archive.

Conversation excerpt IDs: C040, C041, C047.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.

## Supplementary laptop analyses

- [independent-reader-comparison.json](../experiments/evidence/supplementary/independent-reader-comparison.json)
- [independent-reader-publication.json](../experiments/evidence/supplementary/independent-reader-publication.json)
