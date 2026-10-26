"""Arithmetic and missing-data semantics for total utilization charts."""

import math

from render import summed_utilization


def test_total_is_sum_not_average_and_uses_selected_devices():
    samples = [
        {
            "devices": {
                "1": {"aicore": 80, "hbm_used_pct": 50},
                "2": {"aicore": 60, "hbm_used_pct": 40},
                "3": {"aicore": 99},
            }
        }
    ]
    assert summed_utilization(samples, [1, 2], "aicore") == [140]
    assert summed_utilization(samples, [1, 2], "hbm_used_pct") == [90]
    assert summed_utilization(samples, [1, 2, 3], "aicore") == [239]


def test_missing_or_invalid_device_readings_create_gaps():
    for value in (None, float("nan"), float("inf")):
        samples = [{"devices": {"1": {"aicore": 80}, "2": {"aicore": value}}}]
        assert math.isnan(summed_utilization(samples, [1, 2], "aicore")[0])
    assert math.isnan(
        summed_utilization([{"devices": {"1": {"aicore": 80}}}], [1, 2], "aicore")[0]
    )
