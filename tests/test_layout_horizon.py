from areal.trainer.layout_horizon import predict_layout


def choose(missing=0, rate=None, drain=None, terminal=False):
    return predict_layout(
        "split",
        {"split": 40, "borrow": 18},
        {"split": 1, "borrow": 3},
        {"split": 3, "borrow": 8},
        missing,
        rate,
        2,
        drain,
        terminal,
        baseline="split",
        retains_inference={"split": True, "borrow": False},
    )


def test_busy_but_buffered_can_borrow():
    assert choose(drain=2)["world_size"] == "borrow"


def test_data_on_critical_path_preserves_rollout():
    assert choose(25, 1, 2)["world_size"] == "split"


def test_missing_service_or_rate_does_not_imply_zero():
    assert choose()["reason"] == "insufficient_current_run_evidence"
    assert choose(5, None, 2)["world_size"] == "split"


def test_verified_terminal_needs_no_future_publication_or_drain():
    d = choose(terminal=True)
    assert d["world_size"] == "borrow"
    assert d["candidates"]["borrow"]["total"] == 21
