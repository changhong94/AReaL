"""Experiment adapter: official code extraction/tests, explicit E2B execution."""

import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

from kuasar import run_test
from question_rewards import DEFAULT_REWARD_CONCURRENCY

from areal.utils import request_trace
from areal.workflow.openai.code_agent import _build_test_script, _extract_code_block


@contextmanager
def sandbox_slot():
    """One aggregate limit across all rollout processes sharing run-output."""
    directory = Path(os.environ.get("AREAL_CODE_SLOT_DIR", "/run-output/sandbox-slots"))
    directory.mkdir(parents=True, exist_ok=True)
    limit = DEFAULT_REWARD_CONCURRENCY
    start = time.monotonic_ns() % limit
    while True:
        for offset in range(limit):
            stream = (directory / str((start + offset) % limit)).open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
                stream.close()
            return
        start = (start + 1) % limit
        time.sleep(0.01)


def grade(completions, tests, imports, fields, queued_at):
    with sandbox_slot():
        return grade_in_slot(completions, tests, imports, fields, queued_at)


def grade_in_slot(completions, tests, imports, fields, queued_at):
    started = time.time()
    request_trace.emit("code_reward_begin", queued_at=queued_at, **fields)
    try:
        code = _extract_code_block(completions)
        ok, error = run_test(
            _build_test_script(code, tests, imports), trace_fields=fields
        )
    except Exception as exc:
        request_trace.emit("code_reward_error", error_type=type(exc).__name__, **fields)
        raise
    reward = float(ok)
    request_trace.emit(
        "code_reward_end",
        reward=reward,
        seconds=time.time() - started,
        queue_seconds=started - queued_at,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        test_error=error[-1000:],
        **fields,
    )
    return reward
