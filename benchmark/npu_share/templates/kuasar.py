# SPDX-License-Identifier: Apache-2.0
"""Execute code tests through E2B against a configured Kuasar deployment."""

import json
import os
from pathlib import Path


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
        sandbox.files.write("/tmp/areal-test.py", script)
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
