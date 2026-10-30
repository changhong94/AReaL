"""Auxiliary research: cross-process admission to independent sandbox services.

All clients of a pool must share the same local lock directory. This is not a
distributed lock: run one coordinator host, not one independent pool per host.
"""

import fcntl
import hashlib
import json
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

_tls_lock = threading.Lock()
_tls_bundle = None


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    backends = cfg["backends"]
    if not backends or len({b["name"] for b in backends}) != len(backends):
        raise ValueError("Nonempty, uniquely named backends required")
    identities = set()
    for b in backends:
        if not isinstance(b["capacity"], int) or not 1 <= b["capacity"] <= 32:
            raise ValueError(
                "Auxiliary study permits 1..32 configured slots per machine"
            )
        identity = b["machine"]
        if identity in identities:
            raise ValueError("One backend per physical machine required")
        identities.add(identity)
        access = json.loads(Path(b["access_file"]).read_text())
        url = urlsplit(access["api_url"])
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("TLS API or loopback SSH tunnel required")
        if b.get("provider", "kuasar") not in ("kuasar", "cube", "agentenv"):
            raise ValueError("Unsupported sandbox provider")
        b["access"] = access
    if not Path(cfg["lock_dir"]).is_absolute():
        raise ValueError("An absolute, shared local lock_dir is required")
    return cfg


def initialize_tls(cfg):
    """Install one immutable trust bundle; never switch CA per request."""
    global _tls_bundle
    bundle = str(Path(cfg["ca_bundle"]).resolve(strict=True))
    with _tls_lock:
        if _tls_bundle is not None and _tls_bundle != bundle:
            raise RuntimeError("Use one trust bundle per scoring process")
        os.environ["SSL_CERT_FILE"] = bundle
        _tls_bundle = bundle


@contextmanager
def lease(cfg, timeout=300):
    """Choose any free slot, retaining it until sandbox deletion completes."""
    root = Path(cfg["lock_dir"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    slots = [(b, i) for b in cfg["backends"] for i in range(b["capacity"])]
    random.shuffle(slots)
    deadline = time.monotonic() + timeout
    while True:
        for backend, index in slots:
            identity = hashlib.sha256(backend["machine"].encode()).hexdigest()[:16]
            path = root / f"{identity}-{index}"
            with path.open("a+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                # Persist a TTL for crash/ambiguous-create recovery. A dead
                # client's flock disappears but its VM may live until expiry.
                stream.seek(0)
                content = stream.read().strip()
                if content and float(content) > time.time():
                    continue
                state = {"release": False}
                stream.seek(0)
                stream.truncate()
                stream.write(str(time.time() + 180))
                stream.flush()
                try:
                    yield backend, state
                finally:
                    if state["release"]:
                        stream.seek(0)
                        stream.truncate()
                        stream.flush()
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("All sandbox backends are busy or quarantined")
        time.sleep(0.01)


def run_test(script, timeout=10, trace_fields=None, config_path=None):
    from e2b import Sandbox
    from e2b.sandbox.commands.command_handle import CommandExitException

    from areal.utils import request_trace

    cfg = load_config(config_path or os.environ["AREAL_SANDBOX_POOL_CONFIG"])
    if not 1 <= timeout <= 10:
        raise ValueError(
            "The scoring study requires a timeout between 1 and 10 seconds"
        )
    initialize_tls(cfg)
    fields = dict(trace_fields or {})
    queued = time.monotonic()
    with lease(cfg) as (backend, state):
        fields["sandbox_backend"] = backend["name"]
        access = backend["access"]
        sandbox = None
        stage = "create"
        started = time.monotonic()
        request_trace.emit(
            "sandbox_pool_begin", queue_seconds=started - queued, **fields
        )
        try:
            options = {}
            if backend.get("provider", "kuasar") == "kuasar":
                options = {
                    "sandbox_url": access["proxy_url"],
                    "api_headers": {"Host": "api." + access["domain"]},
                }
            elif backend["provider"] == "agentenv":
                options = {"sandbox_url": access["api_url"]}
            sandbox = Sandbox.create(
                access["template_id"],
                timeout=120,
                request_timeout=30,
                api_key=access["api_key"],
                api_url=access["api_url"],
                domain=access.get("domain"),
                **options,
            )
            stage = "upload"
            user_options = {"user": backend["user"]} if backend.get("user") else {}
            sandbox.files.write(
                "/tmp/areal-test.py", script, request_timeout=30, **user_options
            )
            stage = "execute"
            command = (
                f"timeout -k 2s {int(timeout)}s "
                "sh -c 'ulimit -t 2; ulimit -v 524288; "
                "exec python3 /tmp/areal-test.py'"
            )
            try:
                result = sandbox.commands.run(
                    command, timeout=timeout + 15, **user_options
                )
                return result.exit_code == 0, result.stderr
            except CommandExitException as exc:
                return False, exc.stderr
        except Exception as exc:
            request_trace.emit(
                "sandbox_pool_error",
                stage=stage,
                error_type=type(exc).__name__,
                **fields,
            )
            raise
        finally:
            try:
                if sandbox is not None:
                    sandbox.kill(request_timeout=30)
                    state["release"] = True
            finally:
                request_trace.emit(
                    "sandbox_pool_end",
                    seconds=time.monotonic() - started,
                    quarantined=not state["release"],
                    **fields,
                )
