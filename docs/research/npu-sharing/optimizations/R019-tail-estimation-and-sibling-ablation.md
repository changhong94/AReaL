English | [中文](R019-tail-estimation-and-sibling-ablation.zh.md)

# R019: Tail estimation and the rejected sibling-scaling heuristic

Status: Tail/dynamic-service refinements retained; sibling scaling removed; startup
ranking remains offline research.

Reviewed: 2026-10-15 (America/Vancouver). This is a historical evidence review, not a
new benchmark.

## Problem and mechanism

Unfinished answers are censored observations; treating them like completed short answers
underestimates tails. Estimate remaining length conditional on surviving the observed
generated length, with restricted survival means, then simulate completion as active
decode concurrency changes. A later question-specific hazard scale used
completions/exposure to rescale siblings; it was removed because it amplified bias.
Startup prompt-length ranking was separately evaluated offline, not deployed.

## Why the sibling heuristic failed

Among 96 questions, 61 had longest/shortest answer ratio at least two; median ratio was
2.66. More importantly, question 94 had six unfinished answers at 180–184 generated
tokens. Actual mean remaining was 39.83, tail-only prediction 119.24 and sibling-scaled
146.46. The model mistook normal survival for question-wide abnormal length and moved
all siblings together. This does not show that all same-question information is useless.

## Measured prediction and runtime results

Same-input replay of eight decisions: answer-ready MAE old 4.29 s, tail-only 2.91,
tail+sibling 3.31, all refinements 3.18. Removing sibling scaling later improved live
decision MAE 3.73→2.14 s but loop time slightly regressed 154.28→154.98 s with differing
tokens. Separate shared-request replay improved runtime remaining-token MAE 77.46→65.14
(15.9%), while startup MAE only 88.20→86.96 (1.4%) and ranking correlation was about
0.19. Better MAE is not evidence of better ordering or faster training.

## Candidate contribution and next validation

The useful research object is uncertainty-aware residual work estimation under
asynchronous sampling, not the rejected heuristic. Evaluate conditional calibration,
censorship, rank correlation, decision regret and compute overhead on held-out windows.
Preserve the negative sibling experiment as an ablation; do not resurrect it as an
established optimization.

## Code locations

- [areal/infra/rolling_drain.py](../../../../areal/infra/rolling_drain.py)

- [areal/infra/drain_prediction.py](../../../../areal/infra/drain_prediction.py)

## Experiment evidence and conversation provenance

- [refined-drain-comparison01/comparison.json](../experiments/evidence/history/refined-drain-comparison01/comparison.json)

- [refined-drain-comparison01/results-summary.json](../experiments/evidence/history/refined-drain-comparison01/results-summary.json)

- [no-sibling-comparison01/results-summary.json](../experiments/evidence/history/no-sibling-comparison01/results-summary.json)

- [tail-prediction01/summary.json](../experiments/evidence/history/tail-prediction01/summary.json)

Conversation excerpt IDs: C115, C116, C117, C127.
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

- [sibling-diagnosis.json](../experiments/evidence/supplementary/sibling-diagnosis.json)
