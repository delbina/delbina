from delbina import count, cumulative_cost


def test_count_nonzero():
    assert count("hello world") > 0


def test_cumulative_is_triangular():
    turns = ["a b c", "d e f", "g h i"]
    c = [count(t) for t in turns]
    assert cumulative_cost(turns) == c[0] + (c[0] + c[1]) + (c[0] + c[1] + c[2])


def test_empty():
    assert cumulative_cost([]) == 0


def test_counter_presets():
    from delbina import counter
    from delbina.tokens import chars
    assert counter("chars") is chars and chars("abcdefgh") == 2
    assert counter("gpt-4o").__name__ == "tiktoken_o200k_base"
    assert counter("openai/gpt-4.1-mini").__name__ == "tiktoken_o200k_base"
    assert counter("gpt-4").__name__ == "tiktoken_cl100k_base"
    assert counter("claude-sonnet-4-5").__name__ == "estimate_claude"
    assert counter("Qwen2.5-7B").__name__ == "estimate_qwen"
    assert counter("unknown-model") is count


def test_compressor_accepts_model_name():
    from delbina import Compressor
    assert Compressor(count_tokens="chars").count_tokens("abcd") == 1
