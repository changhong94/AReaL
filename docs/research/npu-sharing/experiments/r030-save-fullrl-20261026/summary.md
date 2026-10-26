# R030: full-RL save-path comparison, 2026-10-26

Code RL, FSDP, two shared NPUs, graph inference, 192 questions, 32 questions per batch,
eight answers per question, concurrency 16, scoring concurrency 16, six batches,
offpolicy 1. Initial weights and seed 1 reset before every run. Order: old/new/new/old.
Shared scheduling is identical; four save-related source files are replaced by the
preserved instrumented pre-optimization baseline. Inference engines are reused; training
engines start afresh. Startup is separate.

Before timing, a three-batch DP2/DP1/DP2 correctness run passed five full state imports,
helper gradient relative L2 0.01208594 (existing limit 0.02), and exact readback of 339
saved tensors. All 768 answers were scored without scoring errors. Performance runs
disable full readback/gradient diagnostics as before; each of the four completed 1536
answers and scores without scoring errors.

| Run | Path | RL seconds | Save foreground | Final join | Tokens | Tokens/s |
| --- | ---- | ---------: | --------------: | ---------: | -----: | -------: |
| 1   | old  |    427.202 |          12.158 |      1.927 | 363311 |  850.444 |
| 2   | new  |    422.826 |           3.769 |      1.862 | 358356 |  847.526 |
| 3   | new  |    427.292 |           3.286 |      1.633 | 358476 |  838.949 |
| 4   | old  |    438.011 |          12.562 |      1.956 | 363709 |  830.365 |

Mean foreground saving: 12.360 -> 3.527 s. Final join: 1.942 -> 1.747 s. Combined
critical save time: 14.301 -> 5.275 s, a 9.027 s reduction. Mean RL loop: 432.606 ->
425.059 s, 7.547 s / 1.74% shorter. Mean generated tokens: 363510 -> 358416, 1.40%
fewer. Mean tokens per loop second: 840.404 -> 843.237, only 0.34% higher.

The saving-path benefit repeats, but end-to-end evidence is modest: generation content
differs, earlier PPO/recompute/scoring and entry timings vary, and only two runs per
candidate exist. Do not attribute the entire RL delta to saving. This is a development
warm-session result, not a formal cold-start claim.

The old path's first run directly measured controller serialization 1.085+1.133 s,
worker tokenizer deserialization 3.203/4.277 s, and writer metadata deepcopy 2.966 s.
The two rank timelines overlap. Rank-0 previous-writer collective waited 1.269 s for
rank arrival; the actual snapshot fence was under 1 ms. The new path's first run
captured assets in 0.348 s, used sub-millisecond codec/metadata-copy work, and exposed
0.395/0.642 s DRAM fences that were previously hidden behind metadata work. Independent
CPU weight copy remains required (2.274 s in that new run).

Exact old run timings cannot retrospectively be assigned to the original missing 7.28 s
interval, but the repeated complete pipeline now identifies and removes the expensive
metadata path. No remaining write is excluded from RL duration.

See measurements.json for all stages, input hashes, save events and score metrics.
