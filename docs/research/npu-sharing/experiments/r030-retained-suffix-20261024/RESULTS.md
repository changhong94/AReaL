# R030 retained backing and suffix experiment, 2026-10-24

The requested retained DRAM backing is implemented and fully validated for tracked model
weights and optimizer state. Clean backing copies survive H2D and are reused for
unmapping; dirty generations are asynchronously refreshed and fenced before any
mutation/unmap. KV cache and arbitrary workspaces remain conservatively managed because
their mutation lifecycle is not tracked. The feature remains experimental and opt-in
(`AREAL_RETAIN_DRAM=1`, with the existing idle-training flags). No global defaults were
promoted.

## Results

| Candidate                                  | Fixed loop (s) | Shared loop (s) | Shared slowdown | Shared switch total (s) |
| ------------------------------------------ | -------------: | --------------: | --------------: | ----------------------: |
| Previous mid-batch implementation          |         458.79 |          496.32 |           8.18% |                   68.21 |
| Retained backing + prefetch accounting fix |         456.48 |          484.17 |           6.07% |                   62.52 |
| Retained backing + coalesced DP2 suffix    |         463.37 |          487.96 |           5.31% |                   68.06 |

Each row is a separate development reused-vLLM pair, with a new fixed reference. These
are not matched-answer replays or formal repeated trials. The latest token counts are
fixed 362190, shared 371859; timings include switches, exclude initialization. Neither
new candidate beats fixed. Suffix coalescing has no established end-to-end gain: its
shared loop is 3.79 s slower than the retained-only run, while fixed also varies by 6.89
s. Keep the coalescing code within the opt-in prototype, not as a general performance
claim.

Latest shared PPO excluding nested switching: 232.74 s versus fixed 293.10 s. However
68.06 s of switching plus other shared-path overhead offsets the compute gain.
Generation completion occurs after 66-70 answers have been processed on the single actor
in B2-B5, so expansion uses only the remaining 186-190 answers; actual expansion
opportunities differ between runs.

## Validation and evidence

- Retained candidate: three full batches; nine full rank-local state imports; gradient
  replay relative L2 1.15893%, unchanged 2% threshold.
- Coalesced suffix: three full batches; nine full imports; gradient replay relative L2
  1.15500%. Odd/even suffix CPU tests preserve plan ranges, one optimizer update and
  prefix ownership.
- Targeted suites: 41 backing/state tests, 36 prefetch/pressure tests, 24
  controller/plan/gradient tests passed. Static checks passed on targeted new files.
- Performance: 18 metadata-only imports, no full gradient replay, no sandbox upload
  retries in either arm.
- Latest entire shared process traces (including initialization): 309,418,000,384
  cumulative bytes reused, 188,609,462,272 bytes refreshed in background (11.897 summed
  rank-local seconds), observed backup joins 0.000217 summed seconds. These cumulative
  values do not imply that all copy work is free: producer fences, CPU staging and
  bandwidth contention still contribute elsewhere.
- Initial failed retained attempt: B3 prefetch ignored verified resident workspace, now
  fixed with state-checked incremental budgeting; no safety reserve was reduced.

Latest comparison and four figure families:
`/root/areal-development/idle-retained-suffix-20261024/comparison/20261024-164250-0700_code_fsdp_c16_b6_2b61e5/repetitions/01`
(`figures/`). Retained-only comparison:
`/root/areal-development/idle-retained-v2-20261024/comparison/20261024-153837-0700_code_fsdp_c16_b6_66760f/repetitions/01`.

The external driver completed cleanup and verified `make-npu-busy` running.
Busy-container handling is not part of repository benchmark code. No commit or push
performed.

A next research direction is a measured net-benefit gate for mid-batch expansion, using
remaining original microbatch work and observed round-trip/gradient costs. It is not
implemented or claimed in this experiment.
