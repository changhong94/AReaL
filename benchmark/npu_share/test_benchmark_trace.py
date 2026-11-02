"""Scoring diagnostics must work without research AReaL imports."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def test_cpu_trace_preserves_all_concurrent_events(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_REQUEST_TRACE_DIR", str(tmp_path))
    path = Path(__file__).parent / "templates" / "benchmark_trace.py"
    spec = importlib.util.spec_from_file_location("independent_cpu_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: module.emit("score", request_id=i), range(100)))
    events = [
        json.loads(line)
        for p in tmp_path.glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert sorted(e["request_id"] for e in events) == list(range(100))
    assert all(e["event"] == "score" and e["time"] > 0 for e in events)
