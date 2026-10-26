# R030 overnight optimization results — 2026-10-24

The requested standby staging, overlap and lightweight verification changes are
implemented. Further isolated candidates tested direct snapshot copying, read-only
publication overlap and initialization-only DP2 export prewarming. The latest pair
measured **463.37 s fixed versus 452.65 s shared**, a **10.72 s / 2.31% RL-loop
reduction**, with switching included. Switching decreased from 33.83 s to **15.31 s**
(54.73% lower across these runs).

These are reused-vLLM development comparisons, not independent-engine formal
confirmation or isolated same-generated-input replay. Initialization is outside RL
timing. Final prewarm costs **4.74 s** in initialization. Shared produced **3.69% fewer
tokens** in the final pair, so the whole RL benefit cannot be attributed to code
changes. The unchanged publication-overlap version ranged from a 0.07% slowdown to 1.56%
speedup in two pairs. No claim that performance is maximized or that gains generalize is
justified.

## Conditions and results

Code RL, Qwen2.5-1.5B-Instruct, FSDP, two NPUs, inference TP1, graph inference, six
batches, 32 questions/batch, eight answers/question, question concurrency16, grading
concurrency16, offpolicy1. Fixed retains its original answer grading; shared uses
question-priority grading. Each pair restores initial model state, with separate
fixed/shared service instances retained within the development run.

| Variant                   | Fixed RL (s) | Shared RL (s) | Shared improvement | Shared switch (s) | Fixed / shared tokens |
| ------------------------- | -----------: | ------------: | -----------------: | ----------------: | --------------------: |
| Previous direct backup    |       472.07 |        487.02 |             -3.17% |             33.83 |     372,574 / 360,932 |
| Standby staging + overlap |       457.77 |        470.65 |             -2.81% |             20.41 |     357,213 / 361,736 |
| Direct snapshot copy      |       459.50 |        468.17 |             -1.89% |             20.69 |     361,214 / 356,513 |
| Publication overlap       |       462.24 |        462.56 |             -0.07% |             21.17 |     356,520 / 358,695 |
| Same-code confirmation    |       457.64 |        450.49 |             +1.56% |             17.82 |     362,900 / 351,983 |
| DP2 export prewarm        |       463.37 |        452.65 |             +2.31% |             15.31 |     369,944 / 356,302 |

Positive improvement means shared is faster. Scope timers overlap and must not be summed
as a wall-clock accounting identity.

## Implementation and correctness

- Stage the parked DP2 CPU backup asynchronously after export. Join before snapshot
  overwrite, destination restoration and teardown; validate optimizer/scheduler
  generation and publication version.
- Overlap CPU backup staging with source workspace parking/offload. Restore destination
  HBM only after both complete.
- Cache source mappings and skip unused inventory work when full verification is
  disabled; keep metadata and sampled checks.
- Copy directly into the mapped snapshot and use a private NPU stream. Publish readiness
  only after copy-stream synchronization.
- Poll export completion without blocking the engine RPC thread. Only resident DP1
  full-model fixed-receiver XCCL publication may overlap export. Mutation, save/load,
  slab changes and teardown retain hard fences.
- Prewarm DP2 export during initialization. The real trained generation overwrites the
  initial snapshot; no optimizer update is performed by prewarm.

Eighteen CPU tests passed; generation mismatch, error propagation and overlap contracts
are covered. Every state-path candidate passed full NPU bytewise import validation
before its timed run. The publication-overlap correctness run also observed an actual
publication with copy pending. Performance traces confirm full verification disabled;
metadata/sampled checks remain. No sandbox upload retries were observed in these
completed pairs.

## Latest measurements and remaining limits

Latest first/final DP2 export: 0.755 / 0.748 s. Exposed return snapshot wait: 0.757 /
0.766 s. Return switches: 4.43 / 4.55 s. Four intermediate publications overlap export.
PPO: 294.29 s fixed / 262.37 s shared. Weight-publication scope: 4.51 / 4.34 s. Shared
previous-tail waiting: 5.50 s. Final save: 10.95 / 17.25 s. These numbers suggest
further opportunity in final saving and remaining source export/offload duplication, but
changing them requires a separate correctness/measurement scope; no untested final-batch
shortcut was added.

Latest validated code is retained; R030 remains opt-in through `AREAL_IDLE_TRAINING=1`,
with full verification default on. No commit or push was performed. All completed
overnight experiment roots were moved to `/mnt/nvme1/areal-development/` with their
original `/root/areal-development/` paths retained as symlinks. Nothing was deleted.

## Workspaces and figures

- Previous direct backup:
  `/root/areal-development/idle-direct-20261023/comparison/20261023-235528-0700_code_fsdp_c16_b6_771a75/repetitions/01`
- Standby staging + overlap:
  `/root/areal-development/idle-overlap-20261024/comparison/20261024-012354-0700_code_fsdp_c16_b6_8c4a50/repetitions/01`
- Direct snapshot copy:
  `/root/areal-development/idle-directcopy-20261024/comparison/20261024-022815-0700_code_fsdp_c16_b6_2b4739/repetitions/01`
- Publication overlap:
  `/root/areal-development/idle-publish-overlap-20261024/comparison/20261024-033428-0700_code_fsdp_c16_b6_ee1b57/repetitions/01`
- Same-code confirmation:
  `/root/areal-development/idle-publish-confirm-20261024/comparison/20261024-041150-0700_code_fsdp_c16_b6_774d0b/repetitions/01`
- DP2 export prewarm:
  `/root/areal-development/idle-export-prewarm-20261024/comparison/20261024-051310-0700_code_fsdp_c16_b6_34b69c/repetitions/01`

Latest figures:

`/root/areal-development/idle-export-prewarm-20261024/comparison/20261024-051310-0700_code_fsdp_c16_b6_34b69c/repetitions/01/figures/`

- `2026-10-24_phases/overview.png` and `overview.zh.png`
- `2026-10-24_requests/`
- `2026-10-24_tokens/overview.png`
- `2026-10-24_utilization/overview.png` and `total.png`
