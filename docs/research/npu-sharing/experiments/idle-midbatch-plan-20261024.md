# R030 mid-batch expansion: correctness and six-batch comparison

Completed 2026-10-24 (America/Vancouver). Development warm-vLLM comparison; one pair,
not a formal independent-start result.

## Correctness

Independent repacking changed the numerical layout: frozen-input full-gradient error was
9.96%. Preserving the original complete microbatch packing reduced it to 1.3121%. The
final integrated run passed at 1.2033%, below the unchanged 2% gate, with nine
rank-local full state imports. A separate post-reclaim resident-budget bug was fixed by
re-reading allocator reuse evidence. Relevant test groups passed (22
gradient/plan/controller tests; 31 memory/plan tests). No safety thresholds were
relaxed.

## Performance

Code RL, FSDP, graph inference, two NPUs, 6 batches, 32 questions/batch, 8
answers/question, concurrency 16, reward concurrency 16, offpolicy 1. Fixed retains the
original path; shared enables idle mid-batch expansion. Full replay/byte validation was
disabled for timing; metadata checks remained enabled.

| Metric                               |    Fixed |   Shared |
| ------------------------------------ | -------: | -------: |
| RL loop                              | 458.79 s | 496.32 s |
| PPO scope including nested switch-in | 290.21 s | 267.26 s |
| PPO scope excluding nested switch-in | 290.21 s | 237.42 s |
| Total switch-in and return           |      0 s |  68.21 s |
| Trainer initialization               | 178.77 s | 309.30 s |
| Reported generated tokens            |  361,923 |  361,241 |

Shared is **37.53 s / 8.18% slower**. The 52.79 s reduction in PPO scope excluding
nested handoffs does not cover 68.21 s of switching plus other overheads (including
20.15 s previous-tail waiting). These scopes are nested: do not add total switch time to
the unadjusted PPO scope. PPO scope includes host/RPC overhead and is not pure kernel
time. Tokens differ by only -0.19%, but generated answers still differ; do not claim an
isolated causal comparison of identical training inputs.

B1 and B6 use DP2 at batch entry. B2-B5 switch mid-batch after respectively 34, 68, 70
and 67 of 256 answers; remaining work uses DP2. Switch-in costs are 7.90, 7.34, 7.28 and
7.32 seconds; returns are around 5.3-5.4 seconds. There were no sandbox upload retries.
Eighteen rank-local state imports used metadata checks, with zero diagnostic replay
events. The feature remains opt-in (`AREAL_IDLE_MID_BATCH=1`), not a speedup-default
recommendation.

## Artifacts and cleanup

Workspace:
`/root/areal-development/idle-midbatch-plan-v2-20261024/comparison/20261024-133821-0700_code_fsdp_c16_b6_53cd47/repetitions/01`

Four figure families are under `figures/`: phases (English/Chinese), requests,
utilization, tokens. Detailed scopes and transitions: `timing-analysis.json` in the
parent experiment root. The external driver completed successfully and restored
`make-npu-busy`; busy-container management remains outside the repository
implementation.

Next hypotheses, not implemented: predict sufficient remaining backward work before
expanding; reduce state-return overhead; reduce per-segment RPC/repacking costs without
changing the preserved numerical layout. Do not hide this negative end-to-end result.
