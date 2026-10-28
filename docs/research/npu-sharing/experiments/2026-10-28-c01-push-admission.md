# C01 admission and return-policy repair (2026-10-28)

Coverage remains stopped. No commit or push was performed. Development source changes
are synchronized to hosts 200 and 118.

## Changes

1. Promote the validated event-driven push lease transport to the formal default. Poll
   remains explicit opt-in. Controller and workers agree on the default, and fixed-arm
   generation remains direct.
1. All insufficient-evidence return branches share `uncalibrated_keep`: missing
   cost/service observations, controller observations, request metadata and complete
   forecast horizon. `AREAL_RETURN_CALIBRATION_PROBE=1` explicitly restores exploratory
   returns. The calibrated positive-gain calculation remains unchanged. Without a
   calibration opportunity the conservative policy may remain uncalibrated for the whole
   run; persistent cost reuse is not implemented.
1. The benchmark no longer forces same-device training/inference. Default
   `AREAL_TRAIN_PRIORITY_INFERENCE=0`, explicit1 enables it. This disables simultaneous
   execution, not alternating sharing. Measured-benefit adaptive concurrency is recorded
   as a follow-up, not claimed as implemented.
1. Preserve the requested default unified training workspace and its memory-budget
   safeguards.

## Reused-vLLM paired comparison

Math/FSDP/graph,192 questions,32 questions/batch,eight answers,concurrency16,six
batches,offpolicy2. All shared arms use push. Initial weights are reset between arms.
These are development runs, not cold-start formal confirmation.

| Shared condition    | Fixed seconds | Shared seconds | Reduction |
| ------------------- | ------------: | -------------: | --------: |
| baseline            |        154.94 |         147.80 |     4.60% |
| no-first-return     |        151.42 |         146.08 |     3.52% |
| no-training-overlap |        153.30 |         147.05 |     4.07% |
| both                |        154.44 |         147.44 |     4.53% |

All four runs returned `next_batch_ready` at every return decision. Thus skipping the
first probe was never exercised and must not receive performance credit. Push-only
already removes the previous degradation. Disabling same-card overlap improves the first
training/publication interval but does not establish a significant additional total
benefit. One sample per condition, warm ordering and stochastic outputs remain
confounders.

Original coverage C01:160.382s fixed,194.584s shared. Initial shared native step width
was about21 requests per replica versus64 fixed. With push it reaches62.81/62.92 per
replica over2-15s. Original B2 consumption78.574s; push-only B2 consumption40.240s.
Admission capacity is now utilized; there is no basis to blame the unified workspace for
the old routing slowdown.

## Validation and limits

55 selected routing/decision/runner tests passed after consolidating fallbacks; the
standalone17-test runner suite also passed. Previous full selection including ablations
passed56tests before adding four fallback cases. All eight RL arms completed1536
answers. NPU end-to-end tests used frozen source plus explicit ablation hooks; new
conservative cold fallback branches are covered by unit tests but were not triggered in
these runs. Multinode/TP>1 and all28coverage cases were not rerun.

## Artifacts

Runs:
/root/areal-development/c01-fix-20261028/runs/20261028-012426-0700_math_fsdp_c16_b6_3cac67/repetitions/01
through04. Each has summary and four chart families in figures. comparison.json
preserves decisions and boundaries. Completed checkpoints are archived on
/mnt/nvme2/areal-development/c01-fix-20261028/completed-checkpoints with original-path
symlinks, to avoid exhausting the root filesystem. Logs and evidence are retained.
