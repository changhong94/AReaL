# Incremental budget and handoff timing: four-round comparison and default selection

On 2026-10-20 (America/Vancouver), the user accepted the new version as the default
current implementation. Keep the incremental memory budget and handoff timing fixes in
the working branch; ordinary benchmarks use the current source. The old snapshot remains
a historical control only. This selection is not a claim of demonstrated speedup.

Four development comparison rounds reused vLLM: 96 questions, six answers/question,
concurrency 16, six batches, FSDP, graph. Old mean: 98.996 s; new mean: 99.002 s,
effectively tied. Generated token counts differ, preventing isolated code-overhead
attribution. Training memory safety reserves and same-stream reusable-cache checks
remain. R027 profitability gating remains a separate proposal and is not enabled by this
decision.

| Round | Old seconds | New seconds | Saved seconds | Old tokens | New tokens |
| ----- | ----------: | ----------: | ------------: | ---------: | ---------: |
| 1     |      99.930 |      99.887 |         0.043 |     136642 |     136552 |
| 2     |     100.424 |      98.473 |         1.952 |     138447 |     136124 |
| 3     |      98.633 |      97.441 |         1.191 |     140057 |     134857 |
| 4     |      96.999 |     100.205 |        -3.207 |     133969 |     137444 |

Evidence: [JSON](2026-10-20-budget-old-new-four-rounds.json)
