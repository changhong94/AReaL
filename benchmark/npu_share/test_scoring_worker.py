import importlib.util
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "scoring_worker", Path(__file__).parent / "templates/scoring_worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_duplicate_capacity_and_conflict(tmp_path):
    """Same ID shares execution; capacity rejects new work before acceptance."""
    entered, release = threading.Event(), threading.Event()
    calls = []

    def execute(script, timeout, task_id):
        calls.append(task_id)
        entered.set()
        assert release.wait(5)
        return True, ""

    tasks = worker.Tasks(tmp_path / "tasks.db", 1, execute)
    try:
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(tasks.submit, "one", "pass", 10)
            assert entered.wait(2)
            b = pool.submit(tasks.submit, "one", "pass", 10)
            try:
                with pytest.raises(worker.Rejected) as exc:
                    tasks.submit("two", "pass", 10)
                assert exc.value.status == 429
                with pytest.raises(worker.Rejected) as exc:
                    tasks.submit("one", "different", 10)
                assert exc.value.status == 409
            finally:
                release.set()
            assert a.result() == b.result()
        assert calls == ["one"]
        assert tasks.submit("one", "pass", 10)["passed"] is True
    finally:
        tasks.close()


def test_restart_never_reexecutes_unknown(tmp_path):
    """A durable interrupted record cannot be retried as a fresh task."""
    import hashlib

    path = tmp_path / "tasks.db"
    tasks = worker.Tasks(path, 1, lambda *a: (True, ""))
    digest = hashlib.sha256(json.dumps(["pass", 10]).encode()).hexdigest()
    tasks.db.execute("INSERT INTO tasks VALUES (?,?,NULL)", ("lost", digest))
    tasks.db.commit()
    with pytest.raises(BlockingIOError):
        worker.Tasks(path, 1, lambda *a: pytest.fail("Concurrent instance"))
    tasks.close()
    tasks = worker.Tasks(path, 1, lambda *a: pytest.fail("Duplicate execution"))
    try:
        assert tasks.submit("lost", "pass", 10)["status"] == "unknown"
    finally:
        tasks.close()


def test_infrastructure_failure_not_zero_score(tmp_path):
    """Infrastructure failure differs from an ordinary failing test."""

    def execute(*args):
        raise ConnectionError("private service details")

    tasks = worker.Tasks(tmp_path / "tasks.db", 1, execute)
    try:
        result = tasks.submit("one", "pass", 10)
        assert result["status"] == "error"
        assert "passed" not in result
        assert result["cleanup_confirmed"] is False
        assert "private" not in json.dumps(result)
    finally:
        tasks.close()


def test_http_auth_result_query_and_failed_test(tmp_path):
    """Authenticated submission/query preserve false test results and IDs."""
    tasks = worker.Tasks(tmp_path / "tasks.db", 1, lambda *a: (False, "assertion"))
    server = worker.make_server(("127.0.0.1", 0), tasks, "test-token")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(url + "/v1/tasks/task")
        assert exc.value.code == 401
        req = urllib.request.Request(
            url + "/v1/score",
            data=json.dumps({"task_id": "task", "script": "assert False"}).encode(),
            headers={"Authorization": "Bearer test-token"},
        )
        with opener.open(req) as response:
            result = json.load(response)
        assert result["passed"] is False
        assert result["cleanup_confirmed"] is True
        req = urllib.request.Request(
            url + "/v1/tasks/task", headers={"Authorization": "Bearer test-token"}
        )
        with opener.open(req) as response:
            assert json.load(response) == result
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        tasks.close()


@pytest.mark.parametrize("passed", [True, False])
def test_pool_client_uses_one_remote_lifecycle(tmp_path, monkeypatch, passed):
    """Pool integration returns a worker result without a client-side sandbox."""
    import sys
    import types

    source = Path(__file__).parent / "templates/sandbox_pool.py"
    spec = importlib.util.spec_from_file_location("worker_client_pool", source)
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    calls = []

    def execute(script, timeout, task_id):
        calls.append((script, timeout, task_id))
        return passed, "result"

    tasks = worker.Tasks(tmp_path / "tasks.db", 1, execute)
    server = worker.make_server(("127.0.0.1", 0), tasks, "token")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    monkeypatch.setitem(sys.modules, "e2b", types.SimpleNamespace(Sandbox=None))
    monkeypatch.setitem(
        sys.modules,
        "e2b.sandbox.commands.command_handle",
        types.SimpleNamespace(CommandExitException=Exception),
    )
    events = []
    monkeypatch.setitem(
        sys.modules,
        "areal.utils",
        types.SimpleNamespace(
            request_trace=types.SimpleNamespace(
                emit=lambda name, **kw: events.append((name, kw))
            )
        ),
    )
    cfg = {
        "lock_dir": str(tmp_path / "slots"),
        "backends": [
            {
                "name": "worker",
                "machine": "one",
                "capacity": 1,
                "provider": "agentenv-worker",
                "access": {
                    "api_url": f"http://127.0.0.1:{server.server_port}",
                    "api_key": "token",
                },
            }
        ],
    }
    monkeypatch.setattr(pool, "load_config", lambda _: cfg)
    monkeypatch.setattr(pool, "initialize_tls", lambda _: None)
    try:
        assert pool.run_test("script", config_path="test") == (passed, "result")
        assert len(calls) == 1
        assert calls[0][:2] == ("script", 10)
        assert events[-1][1]["quarantined"] is False
        assert events[-1][1]["worker_seconds"] >= 0
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        tasks.close()
