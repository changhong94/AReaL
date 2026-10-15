English | [中文](R020-tp-tail-negative-result.zh.md)

# R020: Prewarmed TP=2 for remaining long-tail requests

Status: Measured inference-only alternative; dynamic TP/KV migration was not
implemented.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Question and method

Could two devices jointly decode the final few requests faster than a single device?
Replay the same 34 tail requests and exactly 3299 remaining tokens on prewarmed TP=1 and
TP=2, plus one final 275-token request. Repeat five times per condition with
TP1-before/TP2/TP1-after ordering; exclude model initialization and warmup.

## Measured outcome

Combined TP1 median for the group was 16.35 s versus TP2 19.69 s: TP2 was 20.42% slower.
Single tail was 15.09 versus 18.32 s, 21.38% slower. Reserved card-seconds were about
2.41–2.43x. Even before deployment or KV migration costs, this small-model workload
showed no benefit.

## Interpretation and difficulty

Sharding can add communication and smaller kernels; this experiment does not isolate
each cause. DP increases independent-request capacity but cannot divide one
autoregressive dependency chain. Repartitioning an existing request’s KV into TP shards
adds another unimplemented problem. The result is specific to this model, hardware and
request shape, not a universal rejection of TP.

## Next validation

Only revisit with larger models or evidence of compute-limited single-request decoding.
Keep deployment, warmup, redistribution and steady-state costs separate. This negative
design-space result belongs in the paper record, not the optimization success list.

## Code locations

Experiment scripts only; no production scheduler integration. See retained archives.

## Experiment evidence and conversation provenance

- [tp-tail01/summary.json](../experiments/evidence/history/tp-tail01/summary.json)

Conversation excerpt IDs: C124.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
