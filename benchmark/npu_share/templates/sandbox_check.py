"""Check create/execute/delete before reserving NPU resources."""

from kuasar import run_test

ok, error = run_test('assert 2 + 2 == 4\nprint("BENCHMARK_SANDBOX_OK")')
if not ok:
    raise RuntimeError("Sandbox preflight did not pass: " + error[-200:])
print("E2B sandbox create, execute and delete passed")
