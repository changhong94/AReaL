import os


def install_core_reset():
    from vllm.v1.engine.core import EngineCore

    def benchmark_idle(self, payload):
        scheduler = self.scheduler
        if scheduler.requests or scheduler.running or scheduler.waiting:
            raise RuntimeError("Reusable engine must have no requests before reset")
        if getattr(self, "_areal_kv_imports", {}) or getattr(
            scheduler, "_areal_kv_frozen", {}
        ):
            raise RuntimeError("Outstanding KV ownership prevents reset")
        self._areal_kv_import_results = {}
        scheduler._areal_ever_scheduled = set()
        os.environ["AREAL_REQUEST_TRACE_DIR"] = payload["trace_dir"]
        return dict(idle=True, core_pid=os.getpid())

    EngineCore.benchmark_idle = benchmark_idle


install_core_reset()
