"""Development-only adapter; the formal benchmark entrypoint stays unchanged."""

import json
import os

import entry

from areal.api.io_struct import LocalInfServerInfo
from areal.infra.controller import rollout_controller as routing_controller

if os.getenv("AREAL_BENCH_SERVERS"):
    original_initialize = routing_controller.RolloutController.initialize

    def attach_reusable(self, *args, **kwargs):
        if kwargs.get("role", "rollout") == "rollout" and not kwargs.get(
            "server_infos"
        ):
            kwargs["server_infos"] = [
                LocalInfServerInfo(host=row["host"], port=row["port"])
                for row in json.loads(os.environ["AREAL_BENCH_SERVERS"])
            ]
        return original_initialize(self, *args, **kwargs)

    routing_controller.RolloutController.initialize = attach_reusable


if __name__ == "__main__":
    entry.main()
