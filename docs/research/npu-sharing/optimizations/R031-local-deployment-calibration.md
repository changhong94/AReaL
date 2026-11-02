# R031: host-local deployment calibration and measured-cost replacement

Track calibration independently of R030 (training layout) and R027 (inference return).
Reuse only compatible costs on the same machine, never policy winners or model state.
Card numbers are interchangeable; model content, training and rollout parallelism,
runtime and workload are not. Per-layout actual measurements supersede probe values for
the same metric. Immutable revision history and run manifests link each estimate to its
origin; invalidation and rollback require a reason. Measure cold commissioning, warm
lookup, forecast error and end-to-end time separately. See the
[detailed protocol and commands](R031-local-deployment-calibration.zh.md).
