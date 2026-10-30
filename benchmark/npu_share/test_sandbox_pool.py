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
