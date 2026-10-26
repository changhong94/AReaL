"""Bound request chart output without dropping short-run requests."""

from render import request_windows


def test_short_runs_keep_every_request_exactly_once():
    for count in (0, 1, 511, 512, 513, 1024, 1535, 1536):
        windows = request_windows(count)
        selected = [i for _, start, stop in windows for i in range(start, stop)]
        assert selected == list(range(count))
        assert len(windows) <= 3
        assert all(stop - start <= 512 for _, start, stop in windows)


def test_large_runs_select_first_middle_last_without_overlap():
    for count in (1537, 2000, 25600, 100000):
        windows = request_windows(count)
        assert [name for name, _, _ in windows] == ["first", "middle", "last"]
        assert windows[0][1:] == (0, 512)
        assert windows[1][1:] == ((count - 512) // 2, (count - 512) // 2 + 512)
        assert windows[2][1:] == (count - 512, count)
        selected = [i for _, start, stop in windows for i in range(start, stop)]
        assert len(selected) == len(set(selected)) == 1536
