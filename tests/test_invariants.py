"""Randomized checks of the guarantees, across many chats and settings."""
import copy
import json
import random

from delbina import Compressor, is_tool_result
from delbina.messages import fingerprint

words = lambda s: len(s.split())


def random_chat(rng, steps):
    msgs = [{"role": "system", "content": "sys " * rng.randint(1, 30)}]
    for i in range(steps):
        msgs.append({"role": "user", "content": "u " * rng.randint(1, 60)})
        for _ in range(rng.randint(0, 3)):
            cid = f"c{i}_{rng.random():.6f}"
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "f", "arguments": json.dumps({"x": i})}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": "\n".join("t " * 10 for _ in range(rng.randint(1, 80)))})
        msgs.append({"role": "assistant", "content": "a " * rng.randint(1, 80)})
    return msgs


def test_invariants():
    rng = random.Random(0)
    for trial in range(40):
        chat = random_chat(rng, rng.randint(5, 40))
        comp = Compressor(
            max_tokens=rng.choice([300, 800, 2000, 5000]),
            keep_last=rng.randint(0, 6),
            count_tokens=words,
            target_ratio=rng.choice([0.4, 0.6, 1.0]),
            tool_max_tokens=rng.choice([None, 20, 200]),
            keep_tool_results=rng.randint(0, 3),
            summarize=rng.choice([None, lambda lines: "s " * 5]),
            summary_tokens=10,
        )
        prev = None
        for end in range(1, len(chat) + 1):
            if chat[end - 1]["role"] not in ("user", "tool"):
                continue
            msgs = chat[:end]
            snapshot = copy.deepcopy(msgs)
            out = comp.compress(msgs)
            s = comp.last_stats
            assert msgs == snapshot                                   # input untouched
            assert out[0] == msgs[0]                                  # system kept
            assert s.tokens_after == comp.tokens(out)                 # stats are exact
            body = [m for m in out if m["role"] != "system"]
            if s.dropped:
                assert body[0]["role"] == "user" and not is_tool_result(body[0])  # safe cut
            # every kept tool result still has its tool call
            ids = {c["id"] for m in body for c in m.get("tool_calls") or ()}
            assert all(m["tool_call_id"] in ids for m in body if m["role"] == "tool")
            # the last message is always the newest one (maybe trimmed)
            assert out[-1].get("tool_call_id") == msgs[-1].get("tool_call_id")
            if prev is not None and not s.compacted:                  # cache-friendly: prefix unchanged
                assert [fingerprint(m) for m in out[: len(prev)]] == [fingerprint(m) for m in prev]
            prev = out
