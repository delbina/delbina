"""Compaction events: emitted once per compaction, counts only, never secrets or text."""
import logging

from delbina import CompactionEvent, Compressor

words = lambda s: len(s.split())
SECRET = "AKIAIOSFODNN7EXAMPLE"


def chat(n=30):
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": f"TASK use key {SECRET} on port 8443"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": "a " * 30}, {"role": "user", "content": f"u{i} " * 30}]
    return msgs


def run(**kw):
    events = []
    comp = Compressor(max_tokens=400, count_tokens=words, summary_tokens=20, on_compact=events.append, **kw)
    msgs = chat()
    for end in range(2, len(msgs) + 1, 2):
        comp.compress(msgs[:end])
    return comp, events


def test_one_event_per_compaction():
    comp, events = run()
    assert events and all(isinstance(e, CompactionEvent) for e in events)
    assert comp.last_event is events[-1]
    assert all(e.tokens_after <= 400 and e.summary_ms is None and not e.summarized for e in events)


def test_event_fields_with_summary_and_redaction():
    _, events = run(summarize=lambda lines: "kept " + " ".join(w for l in lines for w in l.split() if "SECRET" in w),
                    redact=True, protect="first_user")
    first = events[0]
    assert first.summary_ms is not None and first.pinned == 1
    assert all(not e.redactions for e in events)              # pinned task never reaches the summarizer
    _, events = run(summarize=lambda lines: "kept " + " ".join(w for l in lines for w in l.split() if "SECRET" in w),
                    redact=True)
    assert events[0].redactions == {"aws_access_key": 1} and events[0].summarized


def test_log_has_no_secrets_or_text(caplog):
    with caplog.at_level(logging.INFO, logger="delbina"):
        run(summarize=lambda lines: " ".join(lines), redact=True)
    assert caplog.records and all(r.name == "delbina" for r in caplog.records)
    assert SECRET not in caplog.text and "TASK" not in caplog.text and "8443" not in caplog.text
    assert "compaction:" in caplog.text and "aws_access_key:1" in caplog.text


def test_bad_callback_does_not_break_compression(caplog):
    def boom(event):
        raise RuntimeError("x")
    comp = Compressor(max_tokens=400, count_tokens=words, on_compact=boom)
    out = comp.compress(chat())
    assert comp.tokens(out) <= 400 and "on_compact callback failed" in caplog.text


def test_no_event_without_compaction():
    events = []
    comp = Compressor(max_tokens=10_000, count_tokens=words, on_compact=events.append)
    comp.compress(chat(3))
    assert events == [] and comp.last_event is None
