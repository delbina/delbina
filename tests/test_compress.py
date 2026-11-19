from delbina import cost, cumulative_cost, full, head_tail, last_n, rolling_summary

TURNS = [f"turn {i} " + "word " * 20 for i in range(10)]


def test_full_matches_baseline():
    assert cost(TURNS, full) == cumulative_cost(TURNS)


def test_last_n_keeps_tail():
    assert last_n(3)(TURNS) == TURNS[-3:]


def test_head_tail_keeps_both_ends():
    out = head_tail(1, 2)(TURNS)
    assert out == [TURNS[0]] + TURNS[-2:]


def test_short_history_untouched():
    assert rolling_summary(keep=5)(TURNS[:3]) == TURNS[:3]


def test_compression_saves_tokens():
    for s in (last_n(3), head_tail(1, 2), rolling_summary(keep=2)):
        assert cost(TURNS, s) < cost(TURNS, full)
