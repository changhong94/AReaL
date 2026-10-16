"""Single-turn code RL using the same inference/training path as math RL."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import partial

from code_reward import grade

from areal.utils import request_trace
from areal.utils.perf_tracer import trace_session
from areal.workflow.rlvr import RLVRWorkflow

_request_id = ContextVar("code_request_id")
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kuasar-reward")


class CodeRLVRWorkflow(RLVRWorkflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Prevent the base class from creating its generic reward process pool.
        # _compute_rewards below provides the bounded, strict sandbox executor.
        self.reward_fn = grade

    async def _collect_samples(self, engine, req, prompt_str, task_data):
        token = _request_id.set(req.rid)
        try:
            return await super()._collect_samples(engine, req, prompt_str, task_data)
        finally:
            _request_id.reset(token)

    @trace_session("reward")
    async def _compute_rewards(self, resp, prompt_str, task_data):
        fields = request_trace.identity(_request_id.get())
        completions = self.tokenizer.decode(
            resp.output_tokens, skip_special_tokens=True
        )
        # Queue waiting is unbounded by the 15-second default reward wrapper;
        # each SDK operation and guest process instead has its own deadline.
        # Infrastructure failures propagate; only failed code tests score zero.
        call = partial(
            grade,
            completions,
            task_data["tests"],
            task_data.get("imports", ""),
            fields,
            time.time(),
        )
        return await asyncio.get_running_loop().run_in_executor(_executor, call)
