from delbina import count, cumulative_cost


def test_count_nonzero():
    assert count("hello world") > 0


def test_cumulative_is_triangular():
    turns = ["a b c", "d e f", "g h i"]
    c = [count(t) for t in turns]
    assert cumulative_cost(turns) == c[0] + (c[0] + c[1]) + (c[0] + c[1] + c[2])


def test_empty():
    assert cumulative_cost([]) == 0
