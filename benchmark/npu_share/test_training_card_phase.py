"""Training participation must follow entry selection and handoff gaps."""

from render import training_card_spans


def test_training_cards_follow_selection_and_exclude_handoff():
    data = dict(
        events=[dict(event="idle_training_selected", time=0, batch=1, world_size=1)],
        stages=[
            dict(name="recompute_logp", start=1, end=2),
            dict(name="ppo_update", start=3, end=10),
            dict(name="idle_switch_in", start=5, end=7),
        ],
    )
    result = training_card_spans(data, [1, 2], 1)
    assert [(s["start"], s["end"], s["world"]) for s in result[1]] == [
        (1, 5, 1),
        (7, 10, 2),
    ]
    assert [(s["start"], s["end"]) for s in result[2]] == [(7, 10)]


def test_fixed_training_uses_only_configured_training_cards():
    data = dict(events=[], stages=[dict(name="ppo_update", start=2, end=4)])
    result = training_card_spans(data, [4, 5, 6, 7], 2)
    assert result[4][0]["world"] == 2
    assert result[5] and not result[6] and not result[7]


def test_direct_dual_entry_followed_by_single_entry():
    data = dict(
        events=[
            dict(event="idle_training_selected", time=0, batch=1, world_size=2),
            dict(event="idle_training_selected", time=5, batch=2, world_size=1),
        ],
        stages=[
            dict(name="ppo_update", start=1, end=4),
            dict(name="ppo_update", start=6, end=9),
        ],
    )
    result = training_card_spans(data, [1, 2], 1)
    assert [s["batch"] for s in result[1]] == [1, 2]
    assert [s["batch"] for s in result[2]] == [1]


def test_helper_service_is_separate_from_dual_training():
    data = dict(
        base=100,
        events=[
            dict(event="idle_helper_begin", token="x", batch=1, time=102),
            dict(event="idle_helper_end", token="x", batch=1, time=105),
        ],
        stages=[dict(name="ppo_update", start=1, end=7)],
    )
    result = training_card_spans(data, [1, 2], 1)
    assert result[1][0]["world"] == 1
    assert result[2] == [dict(start=2, end=5, batch=1, world=1, helper=True)]
