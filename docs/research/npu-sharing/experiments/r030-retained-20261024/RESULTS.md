# Retained DRAM backing: six-batch comparison

- Workload: code RL, FSDP, six batches, 32 questions/batch, eight answers/question,
  question concurrency 16, scoring concurrency 16, offpolicy 1, graph inference, NPUs
  1/2.
- Development reused-vLLM run; one fixed/shared pair, not a formal repeated significance
  test.
- Fixed loop: 456.480733 s. Shared loop: 484.171510 s. Shared slower by 27.690778 s
  (6.066%).
- Shared summed switch-in/return: 62.518451 s; prior mid-batch candidate 68.209008 s.
  Prior shared loop 496.320186 s, but run-to-run generation differs, so this is
  contextual evidence rather than an isolated causal estimate.
- PPO scopes: fixed 290.088308 s; shared 264.992539 s including 26.729148 s nested
  mid-batch switches. Shared PPO excluding nested switches: 238.263391 s. Do not add
  nested scopes twice.
- Reported generated tokens: fixed 361000; shared 364037 (+0.84%).
- Full verification OFF in performance run: 18 metadata-only rank-local imports, no
  diagnostic gradient replay. Sandbox upload retries: zero in both arms.
- Initialization (excluded from loop): fixed 180.705436 s, shared 307.683273 s.
- Correctness preceding the run: nine full state imports; B2 full gradient replay
  relative L2 1.15893%; raw retained backing equality passed. Targeted tests: 41
  backing/state tests and 36 prefetch/pressure tests.
- Outcome: mechanism works and reduces copying, but overall performance remains worse
  than fixed in this sample. Keep experimental/opt-in. A subsequent candidate coalesces
  the DP2 suffix, preserving original microbatches and one optimizer step; its outcome
  must be reported separately.

Workspace:
`/root/areal-development/idle-retained-v2-20261024/comparison/20261024-153837-0700_code_fsdp_c16_b6_66760f/repetitions/01`.
Figures: workspace `figures/`.
