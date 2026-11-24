"""protect=: pinned messages survive compaction verbatim, cache-friendly."""
import copy
import random

import pytest

from delbina import BudgetError, Compressor
from delbina.messages import SUMMARY_PREFIX, fingerprint
from test_invariants import random_chat

words = lambda s: len(s.split())


def chat(n=30):
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "TASK: deploy on port 8443 " + "x " * 20}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "a " * 30})
        msgs.append({"role": "user", "content": f"u{i} " * 30})
    return msgs


def test_first_user_survives_and_skips_summarizer():
    seen = []
    comp = Compressor(max_tokens=300, count_tokens=words, protect="first_user",
                      summarize=lambda lines: seen.extend(lines) or "s", summary_tokens=10)
    msgs = chat()
    out = comp.compress(msgs)
    assert comp.last_stats.dropped > 0 and comp.last_stats.pinned == 1
    assert out[1]["content"].startswith(SUMMARY_PREFIX)
    assert out[2] == msgs[1]                                  # pinned, right after the summary
    assert not any("TASK" in line for line in seen)           # not re-summarized
    assert comp.last_stats.tokens_after == comp.tokens(out) <= 300


def test_without_protect_task_is_dropped():
    out = Compressor(max_tokens=300, count_tokens=words).compress(chat())
    assert not any("TASK" in str(m["content"]) for m in out)


def test_protecting_tool_result_keeps_its_call_and_is_not_trimmed():
    big = "line " * 500
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "k", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "k", "content": "KEY=abc " + big}]
    msgs += chat(30)[2:]
    comp = Compressor(max_tokens=900, count_tokens=words, tool_max_tokens=20,
                      protect=lambda m, i: m.get("tool_call_id") == "k")
    out = comp.compress(msgs)
    assert msgs[2] in out and msgs[3] in out                  # call pinned with its result
    assert out.index(msgs[2]) + 1 == out.index(msgs[3])


def test_budget_error_and_bad_preset():
    with pytest.raises(BudgetError):
        Compressor(max_tokens=30, count_tokens=words, protect="first_user").compress(chat())
    with pytest.raises(ValueError):
        Compressor(protect={"nope"})


def test_invariants_with_protect():
    rng = random.Random(1)
    for _ in range(30):
        msgs_all = random_chat(rng, rng.randint(5, 30))
        pick = rng.random()
        comp = Compressor(
            max_tokens=rng.choice([800, 2000, 5000]), keep_last=rng.randint(0, 6), count_tokens=words,
            target_ratio=rng.choice([0.4, 0.6, 1.0]), tool_max_tokens=rng.choice([None, 20, 200]),
            summarize=rng.choice([None, lambda lines: "s " * 5]), summary_tokens=10,
            protect=rng.choice(["first_user", lambda m, i: (hash((i, pick)) % 7 == 0) and m["role"] == "user"]),
        )
        prev = None
        for end in range(1, len(msgs_all) + 1):
            if msgs_all[end - 1]["role"] not in ("user", "tool"):
                continue
            msgs = msgs_all[:end]
            snap = copy.deepcopy(msgs)
            try:
                out = comp.compress(msgs)
            except BudgetError:
                break
            s = comp.last_stats
            assert msgs == snap and out[0] == msgs[0]
            assert s.tokens_after == comp.tokens(out)
            body = [m for m in out if m["role"] != "system"]
            ids = {c["id"] for m in body for c in m.get("tool_calls") or ()}
            assert all(m["tool_call_id"] in ids for m in body if m["role"] == "tool")
            for i in comp._pins:
                assert msgs[1 + i] in out                     # every pinned message is present
            if prev is not None and not s.compacted:
                assert [fingerprint(m) for m in out[: len(prev)]] == [fingerprint(m) for m in prev]
            prev = out
