# R032: Training quality under NPU sharing

Status: planned; no quality conclusion yet.

Test whether sharing or layout transitions degrade training relative to an independent
fixed baseline. Full restore verification is off by default for both own-engine and
imported state. For this study explicitly set `AREAL_IDLE_FULL_VERIFY=1` in the shared
arm. Archive the effective configuration and check verification events and coverage for
applicable restores. An unexercised transition is not a pass.

First replay identical inputs from the same checkpoint and compare rollout/training log
probabilities, gradients, optimizer steps and updated weights. Exercise own-engine
restore, cross-layout restore and a no-switch control; record actual layouts. Then pair
multiple seeds with matching model, data, scorer, sampling settings, training budget and
held-out evaluation. Track policy version and sample staleness. Compare learning curves
and held-out task scores; estimate baseline variation, predeclare quality margins, and
report paired differences with confidence intervals. Inconclusive intervals or
inadequate samples cannot establish equivalence. Passing restore checks does not
establish training quality; verification-on timings are not default performance results.
