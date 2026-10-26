from types import SimpleNamespace

import pytest

from areal.trainer.colocation_memory import GIB, ColocationMemoryPolicy


@pytest.mark.parametrize("credit,passes", [(4, True), (0, False)])
def test_resident_rechecks_fresh_credits_after_reclaim(monkeypatch, credit, passes):
    monkeypatch.setenv("AREAL_RESIDENT_BUDGET", "1")
    monkeypatch.setenv("AREAL_PRESSURE_MEMORY", "1")
    state = dict(
        free_bytes=17 * GIB,
        total_bytes=60 * GIB,
        restore_bytes=0,
        future_optimizer_bytes=0,
        gradient_bytes=6 * GIB,
        reusable_workspace_bytes=4 * GIB,
        reusable_forward_bytes=2 * GIB,
        supports_partial_reclaim=True,
        workspace_parked_bytes=0,
    )
    calls = []

    def actor(action="status", **kwargs):
        calls.append(("actor", action))
        if action == "release_cache":
            state.update(
                free_bytes=19 * GIB,
                reusable_forward_bytes=0,
                reusable_workspace_bytes=credit * GIB,
            )
        return dict(state)

    def rollout(action="status", **kwargs):
        calls.append(("rollout", action))
        if action == "reclaim":
            state["free_bytes"] = 23 * GIB
        return dict(state)

    config = SimpleNamespace(
        train_workspace_gib=16, safety_margin_gib=2, device_budget_gib=None
    )
    policy = ColocationMemoryPolicy(config, actor, rollout, lambda _: None)
    if passes:
        policy.train(resident=True)
    else:
        with pytest.raises(RuntimeError, match="Insufficient memory budget"):
            policy.train(resident=True)
    assert ("rollout", "reclaim") in calls
    assert ("actor", "restore") not in calls
