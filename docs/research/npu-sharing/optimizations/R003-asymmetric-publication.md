English | [中文](R003-asymmetric-publication.zh.md)

# R003: Asymmetric publication to dedicated and shared inference

Status: Implemented; evolved further in R009 and R010.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Mechanism diagram

![R003: Each inference replica resumes after its own version fence](R003.png)

Dedicated readiness no longer depends on shared readiness.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R003.svg) | [PNG](R003.png)

## Problem and mechanism

Waiting for both inference replicas delays the dedicated card behind the shared card’s
memory switching and CPU snapshot load. Publish to the dedicated replica through HCCL;
after its load ACK and version commit it resumes. The shared replica joins only after
its own snapshot load and acknowledgement. The actor and dedicated replica form the HCCL
group; the colocated inference process is excluded.

## Contribution candidate and difficulty

Expose useful inference before every replica is ready while preserving per-replica
version fences. KV slab must support HCCL initialization and full-weight loading instead
of disabling that path. Atomic versioned snapshot manifests replace an obsolete
readiness notification. Failed shared loads keep that replica closed and fail the
training dispatcher; acknowledged dedicated weights stay valid.

## Measured outcome

In the historical 256-question, 16-update run, synchronous-snapshot sharing took 426.58
s and asymmetric sharing 389.89 s (8.6% less). Training through final publication fell
393.60→356.99 s (9.3%). Dedicated readiness averaged 0.89 s and preceded shared
readiness by 3.96 s; 150 generations completed in those gaps. Fixed
training-through-publication was still faster at 340.56 s. Evaluation and sampled-length
differences prevent attributing the full loop change solely to publication.

## Next validation

Separate HCCL completion, snapshot export, shared memory switch, load and admission. The
historical restriction to one actor rank is superseded only by the explicitly tested
multi-rank paths in R022.

## Code locations

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

- [areal/engine/fsdp_engine.py](../../../../areal/engine/fsdp_engine.py)

- [areal/engine/weight_snapshot.py](../../../../areal/engine/weight_snapshot.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

## Experiment evidence and conversation provenance

- [npu-share-hybrid03/comparison.json](../experiments/evidence/history/npu-share-hybrid03/comparison.json)

- [npu-share-fast01/comparison.json](../experiments/evidence/history/npu-share-fast01/comparison.json)

Conversation excerpt IDs: C020.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
