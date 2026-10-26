"""Reward bars represent overlapping active scoring, not queue wait."""

from render import reward_execution_spans, translate_phase_text


def test_reward_spans_merge_workers_without_including_queue():
    events = [
        dict(event="code_reward_begin", pid=1, request_id="a", time=12, queued_at=5),
        dict(event="code_reward_begin", pid=2, request_id="b", time=13),
        dict(event="code_reward_end", pid=1, request_id="a", time=15),
        dict(event="code_reward_end", pid=2, request_id="b", time=16),
        dict(event="code_reward_begin", pid=1, request_id="c", time=20),
        dict(event="code_reward_end", pid=1, request_id="c", time=22),
    ]
    assert reward_execution_spans(dict(events=events, base=10)) == [[2, 6], [10, 12]]
    assert reward_execution_spans(dict(events=[], base=10)) == []
    assert (
        translate_phase_text("Training switch: single to multi")
        != "Training switch: single to multi"
    )
