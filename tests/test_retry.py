import io
import json
import urllib.error

import pytest

import delbina.ollama as ollama
from delbina.evaluate import chat_completion, make_cases, parse_modes, run


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(content="K7Q2ZP"):
    return _Resp(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


def _http_error(code, body=b"{}", headers=None):
    return urllib.error.HTTPError("http://x", code, "err", headers or {}, io.BytesIO(body))


@pytest.fixture
def fake_net(monkeypatch):
    """Queue of responses/exceptions for urlopen; records request bodies and sleeps."""
    state = {"queue": [], "bodies": [], "sleeps": []}

    def urlopen(req, timeout=None):
        state["bodies"].append(json.loads(req.data))
        item = state["queue"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ollama.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ollama, "_sleep", lambda s: state["sleeps"].append(s))
    return state


def test_retries_429_then_succeeds(fake_net):
    fake_net["queue"] = [_http_error(429), _http_error(503), _ok()]
    out = chat_completion("http://x/v1", "m", [{"role": "user", "content": "q"}], retries=3)
    assert out == "K7Q2ZP"
    assert len(fake_net["sleeps"]) == 2


def test_uses_gemini_retry_delay(fake_net):
    body = b'{"error": {"details": [{"retryDelay": "7s"}]}}'
    fake_net["queue"] = [_http_error(429, body), _ok()]
    chat_completion("http://x/v1", "m", [], retries=1)
    assert fake_net["sleeps"] == [8.0]


def test_gives_up_after_retries_and_skips_4xx(fake_net):
    fake_net["queue"] = [_http_error(429), _http_error(429)]
    with pytest.raises(urllib.error.HTTPError):
        chat_completion("http://x/v1", "m", [], retries=1)
    fake_net["queue"] = [_http_error(401)]
    with pytest.raises(urllib.error.HTTPError):
        chat_completion("http://x/v1", "m", [], retries=5)  # auth errors are not retried
    assert len(fake_net["sleeps"]) == 1


def test_reasoning_effort_sent_only_when_set(fake_net):
    fake_net["queue"] = [_ok(), _ok(), _ok("summary")]
    chat_completion("http://x/v1", "m", [], reasoning_effort="minimal")
    chat_completion("http://x/v1", "m", [])
    ollama.openai_summarizer("m", base_url="http://x/v1", api_key="k", reasoning_effort="low")(["a"])
    assert [b.get("reasoning_effort") for b in fake_net["bodies"]] == ["minimal", None, "low"]


def test_rate_limiter_spaces_calls(monkeypatch):
    sleeps = []
    clock = [100.0]
    monkeypatch.setattr(ollama, "_sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(ollama.time, "monotonic", lambda: clock[0])
    limit = ollama.RateLimiter(rpm=30)  # one call every 2 s
    for _ in range(3):
        limit()
    assert sleeps == [pytest.approx(2.0), pytest.approx(2.0)]
    ollama.RateLimiter(0)()  # no limit: never sleeps
    assert len(sleeps) == 2


def test_parse_modes_and_run_subset():
    assert parse_modes(None, have_summary=False) == ["full", "sliding", "delbina"]
    assert parse_modes("full, delbina", have_summary=False) == ["full", "delbina"]
    with pytest.raises(ValueError):
        parse_modes("full,bogus", have_summary=True)
    with pytest.raises(ValueError):
        parse_modes("summary", have_summary=False)
    cases = make_cases(n=2, steps=8, seed=2)
    scores = run(cases, 600, None, count_tokens=lambda s: len(s.split()), modes=["delbina"])
    assert [s.mode for s in scores] == ["delbina"]


def test_missing_content_is_empty_not_crash(fake_net):
    no_content = _Resp(json.dumps({"choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]}).encode())
    tool = _Resp(json.dumps({"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}).encode())
    fake_net["queue"] = [no_content, tool, _Resp(b'{"choices": []}')]
    seen = []
    for _ in range(3):
        assert chat_completion("http://x/v1", "m", [], on_empty=seen.append) == ""
    assert seen == ["stop", "tool_calls", "no choices"]


def test_report_shows_no_reply_column():
    from delbina.evaluate import report
    cases = make_cases(n=3, steps=8, seed=2)
    scores = run(cases, 600, lambda ctx: "", count_tokens=lambda s: len(s.split()), modes=["full"])
    assert scores[0].empty == 3
    assert "no reply" in report(scores, offline=False)


def test_on_retry_reports_reason(fake_net):
    fake_net["queue"] = [urllib.error.URLError(TimeoutError("slow")), _http_error(429), _ok()]
    seen = []
    chat_completion("http://x/v1", "m", [], retries=2, on_retry=lambda n, w, why: seen.append((n, why)))
    assert seen == [(1, "TimeoutError"), (2, "HTTP 429")]


def test_failed_case_is_skipped_not_fatal():
    cases = make_cases(n=3, steps=8, seed=2)
    calls = {"n": 0}

    def ask(ctx):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("network down")
        return ""

    errors = []
    scores = run(cases, 600, ask, count_tokens=lambda s: len(s.split()), modes=["full"], on_error=errors.append)
    assert scores[0].errors == 1 and sum(scores[0].total.values()) == 2
    assert len(errors) == 1 and "TimeoutError" in errors[0]


def test_error_body_is_in_the_message(fake_net):
    body = b'{"error": {"message": "Missing or invalid Authorization header."}}'
    fake_net["queue"] = [_http_error(400, body)]
    with pytest.raises(urllib.error.HTTPError) as err:
        chat_completion("http://x/v1", "m", [], retries=3)
    assert "Missing or invalid Authorization header" in str(err.value)
    assert fake_net["sleeps"] == []  # 400 is not retried


def test_mode_stops_after_errors_in_a_row():
    cases = make_cases(n=5, steps=8, seed=2)
    calls = {"n": 0}

    def ask(ctx):
        calls["n"] += 1
        raise RuntimeError("bad key")

    errors = []
    scores = run(cases, 600, ask, count_tokens=lambda s: len(s.split()), modes=["full"], on_error=errors.append)
    assert calls["n"] == 2                  # gave up after 2, did not try all 5
    assert scores[0].errors == 5
    assert "skipping the other 3" in errors[-1]


def test_report_shows_fact_in_context():
    from delbina.evaluate import report
    cases = make_cases(n=3, steps=8, seed=2)
    scores = run(cases, 10_000, lambda ctx: "no idea", count_tokens=lambda s: len(s.split()), modes=["full"])
    assert scores[0].in_ctx == 3 and scores[0].accuracy == 0
    assert "in context" in report(scores, offline=False)
