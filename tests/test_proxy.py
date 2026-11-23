"""Proxy tests against a fake upstream on localhost."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from delbina import Compressor
from delbina.proxy import serve, upstream_url

words = lambda s: len(s.split())


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: list = []

    def log_message(self, *a):
        pass

    def _reply(self, status, data, ctype="application/json"):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._reply(200, {"path": self.path, "auth": self.headers.get("Authorization")})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.received.append((self.path, dict(self.headers), body))
        if body.get("model") == "fail":
            return self._reply(429, {"error": "rate limited"})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                chunk = f"data: {{\"i\": {i}}}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            return
        self._reply(200, {"n_messages": len(body["messages"])})


@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    Upstream.received = []
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, args=(0.05,), daemon=True).start()
    upstream = f"http://127.0.0.1:{up.server_address[1]}/v1"
    proxy = serve(upstream, "127.0.0.1", 0,
                  compressor=lambda h: Compressor(max_tokens=200, keep_last=2, count_tokens=words), log=False)
    threading.Thread(target=proxy.serve_forever, args=(0.05,), daemon=True).start()
    yield f"http://127.0.0.1:{proxy.server_address[1]}"
    proxy.shutdown()
    up.shutdown()


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), r.read()


def long_chat(n=40):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "w " * 20})
    return msgs


def test_openai_chat_is_compressed(servers):
    status, headers, raw = post(servers + "/v1/chat/completions",
                                {"model": "m", "messages": long_chat()}, {"Authorization": "Bearer sk-test"})
    assert status == 200
    path, up_headers, body = Upstream.received[-1]
    assert path == "/v1/chat/completions"
    assert len(body["messages"]) < 41 and body["messages"][0]["content"] == "sys"
    assert up_headers["Authorization"] == "Bearer sk-test"   # key passed through untouched
    assert "->" in headers["X-Delbina-Tokens"]


def test_small_chat_passes_unchanged(servers):
    msgs = long_chat(2)
    post(servers + "/v1/chat/completions", {"model": "m", "messages": msgs})
    assert Upstream.received[-1][2]["messages"] == msgs


def test_anthropic_summary_goes_to_system(servers):
    msgs = [m for m in long_chat() if m["role"] != "system"]
    post(servers + "/v1/messages", {"model": "claude", "system": "be nice", "messages": msgs})
    body = Upstream.received[-1][2]
    assert body["system"] == "be nice"            # no summarizer configured: system unchanged
    assert all(m["role"] in ("user", "assistant") for m in body["messages"])
    assert body["messages"][0]["role"] == "user"
    assert len(body["messages"]) < len(msgs)


def test_streaming_passthrough(servers):
    status, headers, raw = post(servers + "/v1/chat/completions",
                                {"model": "m", "stream": True, "messages": long_chat()})
    assert raw.count(b"data:") == 3


def test_upstream_error_status_forwarded(servers):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(servers + "/v1/chat/completions", {"model": "fail", "messages": long_chat()})
    assert e.value.code == 429


def test_get_passthrough_and_stats(servers):
    with urllib.request.urlopen(servers + "/v1/models", timeout=10) as r:
        assert json.loads(r.read())["path"] == "/v1/models"
    post(servers + "/v1/chat/completions", {"model": "m", "messages": long_chat()})
    with urllib.request.urlopen(servers + "/delbina/stats", timeout=10) as r:
        stats = json.loads(r.read())
    assert stats["compressed_requests"] >= 1 and stats["tokens_after"] < stats["tokens_before"]


def test_sessions_keep_prefix_stable(servers):
    msgs = long_chat(40)
    post(servers + "/v1/chat/completions", {"model": "m", "messages": msgs})
    first = Upstream.received[-1][2]["messages"]
    msgs = msgs + [{"role": "user", "content": "next"}]
    post(servers + "/v1/chat/completions", {"model": "m", "messages": msgs})
    second = Upstream.received[-1][2]["messages"]
    assert second[: len(first)] == first


def test_bad_json_still_forwarded(servers):
    req = urllib.request.Request(servers + "/v1/chat/completions", data=b"not json",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError):  # fake upstream fails to parse; proxy must not crash
        urllib.request.urlopen(req, timeout=10)
    post(servers + "/v1/chat/completions", {"model": "m", "messages": long_chat(2)})  # still alive


def test_upstream_url():
    assert upstream_url("https://api.openai.com/v1", "/v1/chat/completions") == "https://api.openai.com/v1/chat/completions"
    assert upstream_url("https://openrouter.ai/api/v1/", "/v1/models") == "https://openrouter.ai/api/v1/models"
    assert upstream_url("http://localhost:11434/v1", "/api/chat") == "http://localhost:11434/api/chat"


def test_anthropic_summary_appended_to_system_blocks():
    from delbina.proxy import Sessions, compress_body
    sessions = Sessions(lambda h: Compressor(max_tokens=200, keep_last=2, count_tokens=words,
                                             summarize=lambda lines: "SUM", summary_tokens=5))
    msgs = [m for m in long_chat() if m["role"] != "system"]
    system = [{"type": "text", "text": "be nice", "cache_control": {"type": "ephemeral"}}]
    body = {"model": "claude", "system": system, "messages": msgs}
    stats = compress_body("anthropic", body, sessions, {})
    assert stats.summarized
    assert body["system"][0] == system[0]
    assert body["system"][1]["text"].endswith("SUM")
    assert all(m["role"] != "system" for m in body["messages"])
