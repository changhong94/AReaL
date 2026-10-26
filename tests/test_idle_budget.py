"""Budget decisions must precede any admission fence or device mutation."""

import pytest

from areal.trainer.idle_budget import SwitchBudget


def test_cold_short_suffix_stays_single():
    b = SwitchBudget(initial_cost=15)
    b.observe(1, 1, 100, 10)
    d = b.decide(1, [100, 100])
    assert not d["accepted"]
    assert d["cost_source"] == "entry_prior"


def test_measured_entry_does_not_charge_obsolete_return():
    b = SwitchBudget()
    b.observe(1, 1, 100, 10)
    b.observe(1, 2, 100, 12)
    b.cost(1, "in", 6)
    b.cost(1, "out", 5)
    d = b.decide(2, [100] * 8)
    assert d["accepted"]
    assert d["net_seconds"] == pytest.approx(26)
    assert d["cost_source"] == "measured_entry"


def test_imbalanced_suffix_not_ideal_halving():
    b = SwitchBudget(initial_cost=1)
    b.observe(1, 1, 100, 10)
    assert not b.decide(1, [1000, 1])["accepted"]


def test_window_expires_and_missing_samples_reject():
    b = SwitchBudget()
    b.observe(1, 1, 100, 10)
    assert b.decide(5, [100] * 8)["reason"] == "insufficient_work_or_samples"


def test_odd_suffix_accounts_for_dummy_round():
    b = SwitchBudget(initial_cost=1)
    b.observe(1, 1, 100, 10)
    d = b.decide(1, [100, 100, 100])
    assert d["dual_seconds"] == pytest.approx(200 * 0.1 * 2 / 1.5)


def test_invalid_prior():
    with pytest.raises(ValueError):
        SwitchBudget(initial_cost=-1)


def test_lockstep_rounds_account_for_alternating_heavy_items():
    from areal.trainer.idle_budget import dual_work

    assert dual_work([1000, 1, 1, 1000]) == 2000


def test_incremental_publication_and_terminal_skip():
    b = SwitchBudget(initial_cost=6)
    b.observe(1, 1, 100, 10)
    b.cost(1, "in", 4)
    samples = [(1, 1, "return", 2), (1, 2, "return", 8)]
    assert b.decide(2, [100] * 8, samples)["switch_seconds"] == 10
    assert b.decide(2, [100] * 8, samples, terminal=True)["switch_seconds"] == 4


def test_entry_measurement_works_without_any_out_sample():
    b = SwitchBudget(initial_cost=15)
    b.observe(1, 1, 100, 10)
    b.cost(1, "in", 2)
    assert b.decide(1, [100] * 8)["entry_seconds"] == 2


def test_helper_partition_keeps_primary_final_item():
    from areal.trainer.idle_budget import helper_plan

    d = helper_plan([10] * 8, 1, 2, 1)
    assert d["accepted"] and 0 < d["helper_microbatches"] < 8
    assert d["predicted_seconds"] < 80


def test_helper_rejects_transfer_dominated_work():
    from areal.trainer.idle_budget import helper_plan

    assert not helper_plan([1, 1], 1, 5, 5)["accepted"]
    assert not helper_plan([1], 1, 0, 0)["accepted"]


def test_stale_publication_cost_is_not_free():
    from areal.trainer.idle_entry import EntryBudget

    entry = EntryBudget()
    entry.observe(1, 2, "return", 8)
    entry.observe(4, 1, "return", 2)
    entry.observe(5, 1, "prepare", 0.2)
    budget = SwitchBudget()
    budget.observe(5, 1, 100, 30)
    decision = budget.decide(5, [25] * 4, entry.cost_samples(5))
    assert decision["incremental_publication_seconds"] == 6
    assert decision["publication_source"] == "last_known_fallback"
    entry.observe(5, 2, "return", 3)
    updated = budget.decide(5, [25] * 4, entry.cost_samples(5))
    assert updated["incremental_publication_seconds"] == 1
    assert updated["publication_source"] == "measured"


def test_helper_can_win_even_when_expansion_is_profitable():
    from areal.trainer.idle_budget import prefer_helper

    assert prefer_helper(
        dict(accepted=True, net_seconds=2), dict(accepted=True, net_seconds=6)
    )
    assert not prefer_helper(
        dict(accepted=True, net_seconds=6), dict(accepted=True, net_seconds=2)
    )
    assert not prefer_helper(
        dict(accepted=True, net_seconds=6), dict(accepted=True, net_seconds=6)
    )
    assert not prefer_helper(dict(accepted=False), dict(accepted=False))
