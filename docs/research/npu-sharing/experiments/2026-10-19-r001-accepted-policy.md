# Accepted moderated migration policy

The accepted shared policy retains five requests per exploratory transaction, two
exploratory attempts per run, and stops admitting probes once observed transaction time
reaches two seconds. This is an admission threshold, not a hard transaction timeout.
Positive-gain plans are separate. The runtime default for KV mode is
`AREAL_KV_REBALANCE_POLICY=exploratory`; KV mode itself remains opt-in.

The training-horizon gate is experimental and disabled by default
(`AREAL_KV_TRAINING_HORIZON_GATE=0`). Enable it explicitly inside the trainer container
for research. It was not promoted because no gate fired in validation.

| Independent-startup run   | Fixed seconds | Shared seconds | Reduction | Confirmed imports |
| ------------------------- | ------------: | -------------: | --------: | ----------------: |
| Accepted moderated policy |       179.232 |        152.847 |    14.72% |                10 |
| Optional timing gate      |       184.221 |        160.067 |    13.11% |                 5 |

Both used 96 GSM8K questions, six answers each, concurrency 16, six batches, FSDP and
off-policy head limit two. Accepted-policy token counts were 137098 fixed and 138424
shared; timing-gate counts were 140370 and 138253. The whole shared-policy improvement
is not an isolated causal effect of KV migration. One comparison does not establish
statistical significance or universal benefit.

The timing gate reduced measured transition waiting only from 0.878 to 0.856 seconds. It
fired zero times. A second exploratory attempt was canceled by the existing
training-ready check after approximately 1.953 seconds. Its readiness forecast was 4.565
seconds with two questions still required. This indicates prediction error rather than
demonstrated benefit from the new gate.

Evidence is in [the evidence directory](evidence/r001-moderated-20261019/). Development
iterations use the separate reusable-vLLM tool; final confirmation uses independently
initialized engines through the original benchmark runner. Reuse tooling is not a
research contribution.
