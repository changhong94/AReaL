English | [中文](R018-retained-pause.zh.md)

# R018: Retained same-card pause with partial KV backup

Status: Implemented experimental alternative; migration remains the requested default.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

Migration wastes prefill and can change the policy version during a logical answer.
Pause with keep semantics after the full batch is ready, preserving requests, old
weights and KV. Keep valid pages in HBM when possible; if pressured, copy only required
valid page groups to RAM. After training restore the old version and finish retained
answers before loading new shared weights. The dedicated engine can publish earlier.

## Difficulty and optimization point

Withdrawing one question early can deadlock if its last required answer is paused, so
this mode waits for a full batch. Preserve block IDs/addresses and never use old KV with
newly overwritten weights. A long retained tail may block shared publication and the
next update. This trades replay cost for old-version drain time; it is not strictly
better than migration.

## Measured outcome

A six-batch comparison measured migration 167.09 s and retained pause 164.82 s (1.36%
less), with 135584 versus 136631 tokens. Retained execution had 52 same-card/version
continuations and no card-1-to-2 migrations. The training run had enough HBM and did not
exercise RAM backup. A separate native test moved 4 MiB of a 16 MiB buffer and verified
byte-exact restoration twice. Do not combine these into a claim that partial offload
improved that RL run.

## Next validation

Repeat under actual pressure and controlled tails, measuring both replay avoided and
publication delayed. Keep this branch of the design documented even when the active
baseline uses migration.

## Code locations

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

- [areal/engine/vllm_ext/dynamic_kv.py](../../../../areal/engine/vllm_ext/dynamic_kv.py)

- [areal/engine/vllm_ext/vllm_worker_extension.py](../../../../areal/engine/vllm_ext/vllm_worker_extension.py)

## Experiment evidence and conversation provenance

- [pause-retain-ab01/comparison.json](../experiments/evidence/history/pause-retain-ab01/comparison.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
