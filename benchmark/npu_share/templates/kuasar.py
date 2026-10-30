# SPDX-License-Identifier: Apache-2.0
"""Execute code tests through E2B against a configured Kuasar deployment."""

import json
import os
import time
from pathlib import Path


def _write_test(sandbox, script, trace_fields):
    """Retry idempotent upload while a fresh sandbox route activates."""
    from e2b.exceptions import FileNotFoundException, TimeoutException

    from areal.utils import request_trace

    delays = (0.2, 0.4, 0.8, 1.6, 2.0)
    for attempt in range(len(delays) + 1):
        try:
            sandbox.files.write("/tmp/areal-test.py", script)
            return
        except (FileNotFoundException, TimeoutException) as exc:
            # A missing guest path or other error is not a route readiness error.
            message = str(exc).strip().lower()
            missing_route = (
                isinstance(exc, FileNotFoundException)
                and message == "sandbox not found"
            )
            activating = (
                isinstance(exc, TimeoutException)
                and message.splitlines()[0] == "sandbox activation failed"
            )
            if not (missing_route or activating):
                raise
            request_trace.emit(
                "code_sandbox_upload_missing",
                sandbox_id=sandbox.sandbox_id,
                error_type=type(exc).__name__,
                attempt=attempt + 1,
                exhausted=attempt == len(delays),
                retry_wait_seconds=delays[attempt] if attempt < len(delays) else 0,
                **trace_fields,
            )
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def run_test(script: str, timeout: int = 10, trace_fields=None) -> tuple[bool, str]:
    """Return test outcome; infrastructure failures propagate instead of scoring zero.

    Each call uses a fresh sandbox. Configuration and credentials remain outside
    the repository, in the JSON file selected by KUASAR_CONFIG.
    """
    from areal.utils import request_trace

    trace_fields = trace_fields or {}
    from e2b import Sandbox
    from e2b.sandbox.commands.command_handle import CommandExitException

    cfg = json.loads(Path(os.environ["KUASAR_CONFIG"]).read_text())
    if "backends" in cfg:
        from sandbox_pool import run_test as run_pool_test

        return run_pool_test(
            script, timeout, trace_fields, config_path=os.environ["KUASAR_CONFIG"]
        )
    # SDK's TLS transport reads this before constructing its HTTP clients.
    os.environ["SSL_CERT_FILE"] = cfg["ca_file"]
    sandbox = Sandbox.create(
        template=cfg["template_id"],
        timeout=120,
        request_timeout=30,
        metadata={"areal_run": os.getenv("AREAL_CODE_RUN_ID", "code-three-preflight")},
        api_key=cfg["api_key"],
        api_url=cfg["api_url"],
        domain=cfg["domain"],
        sandbox_url=cfg["proxy_url"],
        api_headers={"Host": "api." + cfg["domain"]},
    )
    request_trace.emit(
        "code_sandbox_created", sandbox_id=sandbox.sandbox_id, **trace_fields
    )
    try:
        _write_test(sandbox, script, trace_fields)
        # The guest shell enforces a wall-clock deadline; SDK transport has
        # a larger timeout so a test timeout is distinct from service failure.
        command = (
            "timeout -k 2s " + str(int(timeout)) + "s "
            "sh -c 'ulimit -t 2; ulimit -v 524288; "
            "exec python3 /tmp/areal-test.py'"
        )
        try:
            result = sandbox.commands.run(command, timeout=timeout + 15)
            return result.exit_code == 0, result.stderr
        except CommandExitException as exc:
            return False, exc.stderr
    finally:
        sandbox.kill(request_timeout=30)
        request_trace.emit(
            "code_sandbox_deleted", sandbox_id=sandbox.sandbox_id, **trace_fields
        )
