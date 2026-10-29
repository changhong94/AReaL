# Host resource hooks

Configure optional host commands in the machine-local benchmark profile:

```json
{
  "resource_hooks": {
    "before": ["bash", "/absolute/path/to/acquire-resources.sh"],
    "after": ["bash", "/absolute/path/to/release-resources.sh"],
    "timeout": 120
  }
}
```

Commands execute without a shell interpolation layer. Each script receives
`AREAL_BENCH_WORKSPACE` and comma-separated `AREAL_BENCH_DEVICES`. Machine-specific
service names belong in these scripts, outside the repository.

The before hook runs once before checking accelerator availability. The after hook runs
once after both experiment groups exit, before chart/report generation. On failure or
interruption it runs after successful experiment cleanup. If cleanup fails, the after
hook is skipped to avoid starting another workload on occupied devices. A failing hook
marks the benchmark failed and is not retried automatically. Hooks must be idempotent;
the after hook also handles partial before-hook failure. Preparation-only and
render-only commands never execute hooks. Unconfigured hooks are no-ops. Output is saved
to resource-before.log and resource-after.log.

This interface applies to run_bench.py and coverage invocations that use it. The
development reuse runner does not automatically inherit these hooks.
