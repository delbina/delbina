"""Prompt-cache behaviour: the output prefix must stay identical between compaction events."""
import asyncio
import json
import threading
import time

from delbina import Compressor, is_tool_result, trim_tool_message
from delbina.messages import fingerprint

words = lambda s: len(s.split())


def grow(n_turns, size=20):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"q{i} " + "w " * size})
        msgs.append({"role": "assistant", "content": f"a{i} " + "w " * size})
    return msgs


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if fingerprint(x) != fingerprint(y):
            break
        n += 1
    return n


def run(comp, turns, size=20):
    outs = []
    for t in range(1, turns + 1):
        outs.append(comp.compress(grow(t, size)))
    return outs


def test_prefix_stable_between_compactions():
    comp = Compressor(max_tokens=500, keep_last=2, count_tokens=words, target_ratio=0.5)
    outs, events = [], 0
    for t in range(1, 60):
        outs.append(comp.compress(grow(t)))
        events += comp.last_stats.compacted
        assert comp.last_stats.tokens_after <= 500
    # Few compaction events, and every non-event call extends the previous output.
    assert 0 < events < 15
    stable = sum(common_prefix(a, b) == len(a) for a, b in zip(outs, outs[1:]))
    assert stable >= len(outs) - 1 - events


def test_target_ratio_one_is_sliding_window():
    comp = Compressor(max_tokens=500, keep_last=2, count_tokens=words, target_ratio=1.0)
    events = 0
    for t in range(1, 60):
        comp.compress(grow(t))
        events += comp.last_stats.compacted
    assert events > 30  # old behaviour: compacts almost every turn


def test_compacts_to_target():
    comp = Compressor(max_tokens=1000, keep_last=2, count_tokens=words, target_ratio=0.6)
    out = comp.compress(grow(40))
    assert comp.last_stats.compacted
    assert comp.tokens(out) <= 600


def test_edited_history_resets_state():
    comp = Compressor(max_tokens=300, keep_last=2, count_tokens=words)
    msgs = grow(30)
    comp.compress(msgs)
    assert comp.last_stats.dropped > 0
    msgs2 = [dict(m) for m in msgs]
    msgs2[3]["content"] = "edited"
    comp.compress(msgs2)
    assert comp.last_stats.compacted  # recomputed from scratch, no stale reuse


def test_new_conversation_under_budget_passthrough():
    comp = Compressor(max_tokens=300, keep_last=2, count_tokens=words)
    comp.compress(grow(30))
    small = grow(2)
    assert comp.compress(small) == small


def test_summary_called_once_per_event():
    calls = []

    def summarize(lines):
        calls.append(lines)
        return "S" + str(len(calls))

    comp = Compressor(max_tokens=600, keep_last=2, count_tokens=words, summarize=summarize, summary_tokens=10)
    events = 0
    for t in range(1, 50):
        out = comp.compress(grow(t))
        events += comp.last_stats.compacted and comp.last_stats.summarized
    assert len(calls) == events
    assert out[1]["content"].endswith(f"S{len(calls)}")
    assert any(l.startswith("previous summary: S") for l in calls[-1])


# ---------------------------------------------------------------- tool output

def agent_chat(n, tool_size=400):
    msgs = [{"role": "system", "content": "agent"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"task {i}"})
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "\n".join(f"line {j} " + "x " * 8 for j in range(tool_size // 10))})
        msgs.append({"role": "assistant", "content": f"done {i}"})
    return msgs


def test_old_tool_outputs_trimmed_recent_kept():
    msgs = agent_chat(6)
    comp = Compressor(max_tokens=2000, keep_last=4, count_tokens=words, tool_max_tokens=50, keep_tool_results=2)
    out = comp.compress(msgs)
    tools = [m for m in out if m.get("role") == "tool"]
    assert len(tools) == 6                      # trimming alone was enough, nothing dropped
    assert comp.last_stats.dropped == 0
    assert comp.last_stats.trimmed == 4
    assert all("delbina trimmed" in t["content"] for t in tools[:4])
    assert tools[-2:] == [m for m in msgs if m.get("role") == "tool"][-2:]
    assert all(t["tool_call_id"] for t in tools)  # ids preserved
    assert words(tools[0]["content"]) < 80


def test_tool_trim_disabled():
    msgs = agent_chat(6)
    comp = Compressor(max_tokens=2000, keep_last=4, count_tokens=words, tool_max_tokens=None)
    out = comp.compress(msgs)
    assert comp.last_stats.trimmed == 0
    assert comp.last_stats.dropped > 0
    assert all("delbina trimmed" not in str(m.get("content")) for m in out)


def test_never_starts_on_tool_result():
    msgs = agent_chat(30)
    for budget in range(300, 3000, 137):
        out = Compressor(max_tokens=budget, keep_last=3, count_tokens=words).compress(msgs)
        first = out[1]
        assert first["role"] == "user" and not is_tool_result(first), budget


def test_anthropic_tool_result_trimmed_but_user_text_kept():
    big = "\n".join("row " + "y " * 10 for _ in range(100))
    m = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": big},
        {"type": "text", "text": "user words " * 100},
    ]}
    out, removed = trim_tool_message(m, 50, words)
    assert removed > 0
    assert out["content"][0]["tool_use_id"] == "t1"
    assert "delbina trimmed" in out["content"][0]["content"]
    assert out["content"][1] == m["content"][1]
    assert m["content"][0]["content"] == big  # input untouched


# ---------------------------------------------------------------- state

def test_state_roundtrip(tmp_path):
    comp = Compressor(max_tokens=500, keep_last=2, count_tokens=words, summarize=lambda l: "SUM")
    msgs = grow(40)
    out1 = comp.compress(msgs)
    path = tmp_path / "state.json"
    comp.save(path)

    calls = []
    comp2 = Compressor(max_tokens=500, keep_last=2, count_tokens=words, summarize=lambda l: calls.append(1) or "NEW")
    assert comp2.load(path)
    msgs.append({"role": "user", "content": "one more"})
    out2 = comp2.compress(msgs)
    assert not calls                     # summary restored, not recomputed
    assert out2[: len(out1)] == out1     # same prefix as before the restart
    assert comp2.summary == "SUM"


def test_load_missing_file(tmp_path):
    assert Compressor(count_tokens=words).load(tmp_path / "nope.json") is False


# ---------------------------------------------------------------- async / background

def test_acompress_with_async_summarizer():
    async def summarize(lines):
        await asyncio.sleep(0)
        return "ASYNC"

    comp = Compressor(max_tokens=300, keep_last=2, count_tokens=words, summarize=summarize)
    out = asyncio.run(comp.acompress(grow(30)))
    assert out[1]["content"].endswith("ASYNC")


def test_sync_compress_rejects_async_summarizer():
    async def summarize(lines):
        return "x"
    comp = Compressor(max_tokens=300, count_tokens=words, summarize=summarize)
    try:
        comp.compress(grow(30))
    except TypeError:
        return
    raise AssertionError("expected TypeError")


def test_background_prefetch_hides_latency():
    started = threading.Event()

    def slow(lines):
        started.set()
        time.sleep(0.3)
        return "BG"

    comp = Compressor(max_tokens=1000, keep_last=2, count_tokens=words, summarize=slow,
                      summary_tokens=10, background=True, prefetch_ratio=0.8)
    t = 1
    while not started.is_set():
        comp.compress(grow(t))
        t += 1
        assert t < 100
    time.sleep(0.4)                       # summary finishes while the user is "typing"
    while True:
        t0 = time.perf_counter()
        out = comp.compress(grow(t))
        if comp.last_stats.compacted:
            break
        t += 1
    assert time.perf_counter() - t0 < 0.2  # compaction used the prepared summary
    assert out[1]["content"].endswith("BG")
    comp.close()
