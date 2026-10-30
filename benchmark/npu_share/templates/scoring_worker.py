"""AUX-S01 private scoring worker. Generated code executes only inside E2B.

Bind loopback and expose through an authenticated SSH tunnel. The SQLite journal
retains task identities across restarts; interrupted tasks are never rerun.
"""

import argparse
import fcntl
import hashlib
import hmac
import json
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Rejected(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class Tasks:
    def __init__(self, database, capacity, execute):
        if type(capacity) is not int or not 1 <= capacity <= 32:
            raise ValueError("Worker capacity must be 1..32")
        self.instance_lock = open(str(database) + ".lock", "a+")
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.instance_lock.close()
            raise
        self.capacity, self.execute = capacity, execute
        self.condition = threading.Condition()
        self.active = 0
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, digest TEXT, result TEXT)"
        )
        # NULL means execution may have begun before a process crash. Never repeat it.
        self.db.execute(
            "UPDATE tasks SET result=? WHERE result IS NULL",
            (
                json.dumps(
                    {
                        "status": "unknown",
                        "cleanup_confirmed": False,
                        "error_type": "WorkerRestarted",
                        "message": "Interrupted task; no automatic replay",
                    }
                ),
            ),
        )
        self.db.commit()

    def close(self):
        self.db.close()
        self.instance_lock.close()

    def lookup(self, task_id):
        with self.condition:
            row = self.db.execute(
                "SELECT result FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise Rejected(404, "Unknown task")
            return json.loads(row[0]) if row[0] else {"status": "running"}

    def submit(self, task_id, script, timeout):
        if not isinstance(task_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", task_id
        ):
            raise Rejected(400, "Invalid task ID")
        if not isinstance(script, str) or len(script.encode()) > 2 * 1024 * 1024:
            raise Rejected(400, "Invalid or oversized script")
        if type(timeout) is not int or not 1 <= timeout <= 10:
            raise Rejected(400, "Timeout must be 1..10 seconds")
        digest = hashlib.sha256(json.dumps([script, timeout]).encode()).hexdigest()
        with self.condition:
            row = self.db.execute(
                "SELECT digest,result FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row:
                if row[0] != digest:
                    raise Rejected(409, "Task ID reused with different input")
                deadline = time.monotonic() + 150
                while row[1] is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {"status": "running"}
                    self.condition.wait(remaining)
                    row = self.db.execute(
                        "SELECT digest,result FROM tasks WHERE id=?", (task_id,)
                    ).fetchone()
                return json.loads(row[1])
            if self.active >= self.capacity:
                raise Rejected(429, "Worker capacity exhausted; task not accepted")
            self.db.execute("INSERT INTO tasks VALUES (?,?,NULL)", (task_id, digest))
            self.db.commit()
            self.active += 1
        started = time.monotonic()
        try:
            passed, stderr = self.execute(script, timeout, task_id)
            result = {
                "status": "completed",
                "passed": bool(passed),
                "stderr": str(stderr)[-65536:],
                "cleanup_confirmed": True,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "error_type": type(exc).__name__,
                "cleanup_confirmed": False,
            }
        result["worker_seconds"] = time.monotonic() - started
        with self.condition:
            try:
                self.db.execute(
                    "UPDATE tasks SET result=? WHERE id=?",
                    (json.dumps(result), task_id),
                )
                self.db.commit()
            finally:
                self.active -= 1
                self.condition.notify_all()
        return result


def make_server(address, tasks, token):
    if address[0] not in ("127.0.0.1", "localhost") or not token:
        raise ValueError("Loopback binding and a nonempty private token required")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log request bodies, scripts or authentication headers.

        def respond(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Committed result remains available after client disconnect.

        def authorized(self):
            return hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            )

        def do_GET(self):
            if not self.authorized():
                return self.respond(401, {"error": "Unauthorized"})
            try:
                if not self.path.startswith("/v1/tasks/"):
                    raise Rejected(404, "Unknown route")
                self.respond(200, tasks.lookup(self.path[len("/v1/tasks/") :]))
            except Rejected as exc:
                self.respond(exc.status, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return self.respond(401, {"error": "Unauthorized"})
            try:
                if self.path != "/v1/score":
                    raise Rejected(404, "Unknown route")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 3 * 1024 * 1024:
                    raise Rejected(413, "Invalid body size")
                self.connection.settimeout(15)
                data = json.loads(self.rfile.read(size))
                result = tasks.submit(
                    data["task_id"], data["script"], data.get("timeout", 10)
                )
                self.respond(200, result)
            except Rejected as exc:
                self.respond(exc.status, {"error": str(exc)})
            except (ValueError, KeyError, TypeError):
                self.respond(400, {"error": "Invalid request"})

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    args = parser.parse_args()
    from sandbox_pool import load_config, run_test

    cfg = load_config(args.config)
    if len(cfg["backends"]) != 1 or cfg["backends"][0].get("provider") != "agentenv":
        raise ValueError("Worker requires exactly one local AgentEnv backend")
    from urllib.parse import urlsplit

    if urlsplit(cfg["backends"][0]["access"]["api_url"]).hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        raise ValueError("AgentEnv must run on this worker's host")
    if cfg.get("lease_timeout") != 0:
        raise ValueError(
            "Worker local pool requires lease_timeout=0; never hide a queue"
        )
    tasks = Tasks(
        args.database,
        args.capacity,
        lambda script, timeout, task_id: run_test(
            script, timeout, {"worker_task_id": task_id}, args.config
        ),
    )
    server = make_server(
        ("127.0.0.1", args.port), tasks, Path(args.token_file).read_text().strip()
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        tasks.close()


if __name__ == "__main__":
    main()
