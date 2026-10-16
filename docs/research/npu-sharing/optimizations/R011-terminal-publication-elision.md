English | [中文](R011-terminal-publication-elision.zh.md)

# R011: Skip unused terminal inference restoration

Status: Implemented with evaluation-aware finality.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Mechanism diagram

![R011: Eliminate final publication only when nobody needs it](R011.png)

An epoch boundary alone is not finality. Required evaluation still gets fresh weights.

Schematic only: geometry does not encode time or bytes. Status and result boundaries
follow this note.

[SVG](R011.svg) | [PNG](R011.png)

## Problem and mechanism

If training has truly ended and no evaluation needs fresh inference weights, restoring
inference creates work nobody consumes. Determine the actual final optimizer step from
epoch-derived limits and total_train_steps. Skip dedicated publication, shared snapshot,
memory switching and reopening inference; still save requested model/recovery state and
finish statistics/cleanup.

## Difficulty and optimization point

An epoch boundary is not necessarily the end. Due evaluation must retain publication so
it uses trained weights. Already-started background work must finish before destruction.
This is terminal dead-work elimination, an engineering contribution rather than a novel
scheduling algorithm.

## Measured outcome

Later three- and six-batch summaries record terminal_publications_skipped=1. No clean
same-input on/off ablation was recovered, so do not assign a standalone percentage. The
historical independent-reader run intentionally has only two publications for three
updates.

## Next validation

Compare terminal publication on/off under identical inputs, checking final saved weights
and evaluation behavior in both cases.

## Code locations

- [areal/trainer/rl_trainer.py](../../../../areal/trainer/rl_trainer.py)

## Experiment evidence and conversation provenance

- `terminal-tail01`: Logs/scripts are in the persistent historical archive.

- [post-warmup-ab01/comparison.json](../experiments/evidence/history/post-warmup-ab01/comparison.json)

Conversation excerpt IDs: C043.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
