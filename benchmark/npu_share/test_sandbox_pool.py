import importlib.util
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SOURCE = Path(__file__).parent / "templates" / "sandbox_pool.py"
spec = importlib.util.spec_from_file_location("sandbox_pool", SOURCE)
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def config(directory):
    return {
        "lock_dir": directory,
        "backends": [
            {"name": "one", "machine": "host-one", "capacity": 8},
            {"name": "two", "machine": "host-two", "capacity": 8},
        ],
    }


def hold(cfg, ready, release):
    with pool.lease(cfg) as (backend, state):
        ready.put(backend["name"])
        release.wait(10)
        state["release"] = True


def test_full_backend_routes_to_free_backend(tmp_path):
    """A saturated backend never prevents admission to the other backend."""
    import threading

    cfg = config(str(tmp_path))
    one = dict(cfg, backends=cfg["backends"][:1])
    with ThreadPoolExecutor(max_workers=8) as executor:
        barrier = threading.Barrier(9)
        release = threading.Event()

        def worker():
            with pool.lease(one) as (_, state):
                barrier.wait(5)
                release.wait(5)
                state["release"] = True

        futures = [executor.submit(worker) for _ in range(8)]
        try:
            barrier.wait(5)
            with pool.lease(cfg, timeout=1) as (backend, state):
                assert backend["name"] == "two"
                state["release"] = True
        finally:
            release.set()
        for future in futures:
            future.result()


def test_capacity_shared_across_processes(tmp_path):
    """Two processes cannot claim the same machine's only remaining slot."""
    ctx = multiprocessing.get_context("fork")
    cfg = config(str(tmp_path))
    cfg["backends"] = [dict(cfg["backends"][0], capacity=1)]
    ready, release = ctx.Queue(), ctx.Event()
    process = ctx.Process(target=hold, args=(cfg, ready, release))
    process.start()
    try:
        assert ready.get(timeout=5) == "one"
        with pytest.raises(TimeoutError):
            with pool.lease(cfg, timeout=0.03):
                pytest.fail("Over-admission")
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    with pool.lease(cfg, timeout=0.1) as (_, state):
        state["release"] = True


def test_ambiguous_creation_quarantines_slot(tmp_path):
    """Unconfirmed cleanup must not immediately admit an additional VM."""
    cfg = config(str(tmp_path))
    cfg["backends"] = [dict(cfg["backends"][0], capacity=1)]
    with pool.lease(cfg):
        pass
    with pytest.raises(TimeoutError):
        with pool.lease(cfg, timeout=0.03):
            pytest.fail("Quarantined slot reused")
    next(tmp_path.iterdir()).write_text(str(time.time() - 1))
    with pool.lease(cfg, timeout=0.1) as (_, state):
        state["release"] = True


@pytest.mark.parametrize("failure", [None, "create", "delete"])
def test_lifecycle_failure_and_timing(tmp_path, monkeypatch, failure):
    """Infrastructure failures propagate without retry and cleanup is traced."""
    import sys
    import types

    cfg = config(str(tmp_path))
    cfg["backends"] = [
        dict(
            cfg["backends"][0],
            provider="agentenv",
            access=dict(api_key="test", api_url="http://localhost", template_id="test"),
        )
    ]
    calls, events = [], []

    def create(*args, **kwargs):
        calls.append("create")
        if failure == "create":
            raise ConnectionError("creation failed")
        return types.SimpleNamespace(
            files=types.SimpleNamespace(write=lambda *a, **k: None),
            commands=types.SimpleNamespace(
                run=lambda *a, **k: types.SimpleNamespace(exit_code=0, stderr="")
            ),
            kill=kill,
        )

    def kill(**kwargs):
        calls.append("delete")
        if failure == "delete":
            raise ConnectionError("cleanup failed")

    monkeypatch.setitem(
        sys.modules,
        "e2b",
        types.SimpleNamespace(Sandbox=types.SimpleNamespace(create=create)),
    )
    monkeypatch.setitem(
        sys.modules,
        "e2b.sandbox.commands.command_handle",
        types.SimpleNamespace(CommandExitException=type("Exit", (Exception,), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "areal.utils",
        types.SimpleNamespace(
            request_trace=types.SimpleNamespace(
                emit=lambda name, **kw: events.append((name, kw))
            )
        ),
    )
    monkeypatch.setattr(pool, "load_config", lambda _: cfg)
    monkeypatch.setattr(pool, "initialize_tls", lambda _: None)
    if failure:
        with pytest.raises(ConnectionError):
            pool.run_test("pass", config_path="test")
        assert calls.count("create") == 1
        assert any(
            name == "sandbox_pool_error" and data["stage"] == failure
            for name, data in events
        )
    else:
        assert pool.run_test("pass", config_path="test") == (True, "")
        assert calls == ["create", "delete"]
        assert all(
            events[-1][1][key] >= 0
            for key in (
                "create_seconds",
                "upload_seconds",
                "execute_seconds",
                "delete_seconds",
            )
        )
    assert events[-1][1]["quarantined"] == bool(failure)
