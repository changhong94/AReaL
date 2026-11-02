"""Community RLVR generation with only an external CPU scoring adapter."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import partial

from code_reward import grade
from question_rewards import DEFAULT_REWARD_CONCURRENCY

from areal.workflow.rlvr import RLVRWorkflow

_request_id = ContextVar("baseline_code_request_id")
_executor = ThreadPoolExecutor(
    max_workers=DEFAULT_REWARD_CONCURRENCY, thread_name_prefix="baseline-reward"
)


class CodeRLVRWorkflow(RLVRWorkflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_fn = grade

    async def _collect_samples(self, engine, req, prompt_str, task_data):
        token = _request_id.set(req.rid)
        try:
            return await super()._collect_samples(engine, req, prompt_str, task_data)
        finally:
            _request_id.reset(token)

    async def _compute_rewards(self, resp, prompt_str, task_data):
        completion = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
        call = partial(
            grade,
            completion,
            task_data["tests"],
            task_data.get("imports", ""),
            {"request_id": _request_id.get()},
            time.time(),
        )
        return await asyncio.get_running_loop().run_in_executor(_executor, call)
