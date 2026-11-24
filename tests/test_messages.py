import pytest

from delbina import Compressor, compress, message_text

words = lambda s: len(s.split())  # offline token counter for tests


def chat(n, size=10):
    msgs = [{"role": "system", "content": "be helpful"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"m{i} " + "w " * size})
    return msgs


def test_under_budget_unchanged():
    msgs = chat(4)
    assert compress(msgs, max_tokens=10_000, count_tokens=words) == msgs


def test_input_not_modified():
    msgs = chat(20)
    copy = [dict(m) for m in msgs]
    compress(msgs, max_tokens=60, count_tokens=words)
    assert msgs == copy


def test_fits_budget_and_keeps_system_and_tail():
    msgs = chat(20)
    c = Compressor(max_tokens=80, keep_last=4, count_tokens=words)
    out = c.compress(msgs)
    assert out[0] == msgs[0]
    assert out[-4:] == msgs[-4:]
    assert c.tokens(out) <= 80
    assert c.last_stats.dropped > 0 and c.last_stats.saved > 0


def test_keep_last_is_never_dropped_even_over_budget():
    msgs = chat(6, size=50)
    out = compress(msgs, max_tokens=10, keep_last=4, count_tokens=words)
    assert out[-4:] == msgs[-4:]


def test_kept_part_starts_with_user():
    msgs = chat(21)
    out = compress(msgs, max_tokens=90, keep_last=3, count_tokens=words)
    assert out[1]["role"] == "user"


def test_summary_inserted_and_cached():
    calls = []

    def summarize(lines):
        calls.append(list(lines))
        return "S"

    c = Compressor(max_tokens=100, keep_last=2, summarize=summarize, summary_tokens=5, count_tokens=words)
    msgs = chat(20)
    out = c.compress(msgs)
    assert out[1]["content"].endswith("S") and out[1]["role"] == "system"
    assert c.last_stats.summarized

    c.compress(msgs)                       # same chat: cached, no new call
    assert len(calls) == 1

    msgs += [{"role": "user", "content": "new " * 40}, {"role": "assistant", "content": "ok " * 40}]
    c.compress(msgs)                       # grew: only new messages + previous summary
    assert len(calls) == 2
    assert calls[1][0].startswith("previous summary:")
    assert len(calls[1]) < len(calls[0]) + 2


def test_content_blocks():
    m = {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": {}}]}
    assert message_text(m) == "hi"


def test_bad_args():
    with pytest.raises(ValueError):
        Compressor(max_tokens=0, count_tokens=words)


def test_long_summary_is_capped_to_its_reserve():
    rambling = lambda lines: "fact " * 500          # summarizer ignores the length limit
    c = Compressor(max_tokens=100, keep_last=2, summarize=rambling, summary_tokens=20, count_tokens=words)
    out = c.compress(chat(30))
    assert words(c.summary) <= 20
    assert c.last_stats.tokens_after <= 100           # budget holds despite the summarizer
    assert out[1]["content"].startswith("Summary of the earlier conversation:")
