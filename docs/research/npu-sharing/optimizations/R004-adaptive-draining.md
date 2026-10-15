English | [中文](R004-adaptive-draining.zh.md)

# R004: Adaptive draining with a rolling completion-time model

Status: Implemented; three-window policy retained, sibling-length scaling removed.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and evolution

Immediate migration discards reusable KV and re-prefills prefixes; indefinite
old-version draining delays publication and the next PPO. The first compromise was a
five-second cap, followed by event exits (empty, batch ready, version/budget limits) and
a cost heuristic with a 15% margin/two votes. The current batch model removes those
fixed heuristics and freezes observations at each publication boundary.

## Objective and estimation

`J(a) = max(T_answers_ready(a), T_training_card_ready(a))`, for stay versus migrate. The
answer horizon requires complete questions (the slowest of six answers), not independent
answer count. Card readiness includes native pause, publication and actor restoration,
excluding idle waiting. Merge up to three completed batch windows; use all available at
startup, retain provenance for missing metrics, and never replace unknown costs with
zero. Conservatism coefficient starts at zero. Completion events and measured handoff
are causally fenced; no future lengths or batch membership may leak into predictions.

## Refinements and difficulty

CPU pure-decode frame intervals are bucketed by concurrency and context; paused/prefill
intervals are excluded. Empirical execution slots model queued work. Right-censored
lengths and a restricted conditional survival mean improve tail estimates; dynamic TPOT
changes as the live set shrinks. These are estimated service times, not NPU kernel
times. Prediction cost itself (historically about 0.24–0.40 s) can matter. Same-question
hazard scaling was removed after it amplified overprediction (R019).

## Measured results, including regressions

Adaptive run: immediate 159.87 s, five-second drain 153.59 s, early adaptive 158.35 s;
migrations 251/173/58. Fewer migrations did not imply faster training. Batch-window run:
153.93/152.43/150.20 s for fixed-time/early-adaptive/batch policy. Enlarging the window
plus readiness fixes regressed 150.05→155.81 s, with tokens 133449→140660.
Tail/sibling/dynamic refinements later measured 155.52→145.76 s, also with fewer tokens.
These are separate single-run experiments, not an additive speedup series. Same-input
estimator replay and sibling removal are reported separately in R019.

## Paper framing and next validation

Candidate: choose the next training critical path using causally observed batch
statistics rather than a fixed drain timeout. Novelty versus prior
scheduling/survival-analysis work is unverified. Compare every estimator on the same
frozen decision states, then repeat end-to-end paired runs; report readiness MAE,
planning overhead, quality and negative outcomes.

## Code locations

- [areal/infra/adaptive_drain.py](../../../../areal/infra/adaptive_drain.py)

- [areal/infra/batch_drain.py](../../../../areal/infra/batch_drain.py)

- [areal/infra/rolling_drain.py](../../../../areal/infra/rolling_drain.py)

- [areal/infra/drain_prediction.py](../../../../areal/infra/drain_prediction.py)

- [areal/infra/controller/rollout_controller.py](../../../../areal/infra/controller/rollout_controller.py)

## Experiment evidence and conversation provenance

- [bounded-drain-comparison01/comparison.json](../experiments/evidence/history/bounded-drain-comparison01/comparison.json)

- [adaptive-drain-comparison01/comparison.json](../experiments/evidence/history/adaptive-drain-comparison01/comparison.json)

- [adaptive-drain-comparison01/results-summary.json](../experiments/evidence/history/adaptive-drain-comparison01/results-summary.json)

- [batch-drain-comparison02/comparison.json](../experiments/evidence/history/batch-drain-comparison02/comparison.json)

- [batch-drain-comparison02/results-summary.json](../experiments/evidence/history/batch-drain-comparison02/results-summary.json)

- [rolling-drain-comparison01/results-summary.json](../experiments/evidence/history/rolling-drain-comparison01/results-summary.json)

- [refined-drain-comparison01/comparison.json](../experiments/evidence/history/refined-drain-comparison01/comparison.json)

- [refined-drain-comparison01/results-summary.json](../experiments/evidence/history/refined-drain-comparison01/results-summary.json)

- [no-sibling-comparison01/results-summary.json](../experiments/evidence/history/no-sibling-comparison01/results-summary.json)

Conversation excerpt IDs: C105, C107, C110, C113, C115, C117.
[Open excerpts](../experiments/evidence/conversation-excerpts.json)

Raw evidence root: `/root/areal-research-artifacts/history-20261015/`. This is a
selected archive, excluding model checkpoints and full profiler payloads; it is not a
complete runnable backup.

## Publication claim boundary

A contribution candidate identifies a mechanism worth studying, not established novelty;
a systematic related-work review is pending. Live RL can change output lengths,
consumption order and training trajectories. Percentages from different runs are not
additive. Without an isolated ablation, gains belong to the combined system.
