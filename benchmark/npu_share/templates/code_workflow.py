"""Single-turn code RL using the same inference/training path as math RL."""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import partial

from code_reward import grade, grade_in_slot
from question_rewards import DEFAULT_REWARD_CONCURRENCY, QuestionRewardQueue

from areal.infra import workflow_context
from areal.infra.workflow_context import notify_generation_complete
from areal.utils import request_trace
from areal.utils.perf_tracer import trace_session
from areal.workflow.rlvr import RLVRWorkflow

_request_id = ContextVar("code_request_id")
_executor = ThreadPoolExecutor(
    max_workers=DEFAULT_REWARD_CONCURRENCY, thread_name_prefix="kuasar-reward"
)
_question_queues = {}


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
        # This single-turn sample has finished all model generation.
        await notify_generation_complete()
        fields = request_trace.identity(_request_id.get())
        completions = self.tokenizer.decode(
            resp.output_tokens, skip_special_tokens=True
        )
        # Queue waiting is unbounded by the 15-second default reward wrapper;
        # each SDK operation and guest process instead has its own deadline.
        # Infrastructure failures propagate; only failed code tests score zero.
        policy = os.environ.get("AREAL_CODE_REWARD_SCHEDULING", "answer")
        call = partial(
            grade_in_slot if policy == "question" else grade,
            completions,
            task_data["tests"],
            task_data.get("imports", ""),
            fields,
            time.time(),
        )
        if policy == "question":
            context = workflow_context.get()
            loop = asyncio.get_running_loop()
            queue = _question_queues.get(loop)
            if queue is None:
                queue = _question_queues[loop] = QuestionRewardQueue()
            key = context.generation_completion
            if key is None:
                raise RuntimeError(
                    "Question scoring requires a group completion context"
                )
            return await queue.submit(
                key, context.sample_idx or 0, context.group_size, call
            )
        if policy != "answer":
            raise ValueError(f"Unknown code reward scheduling policy: {policy}")
        return await asyncio.get_running_loop().run_in_executor(_executor, call)
