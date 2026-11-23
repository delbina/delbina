"""Drop-in compression proxy.

    delbina serve --upstream https://api.openai.com/v1 --max-tokens 8000

Then point any OpenAI-compatible client at ``http://127.0.0.1:8787/v1``.
Chat requests get their ``messages`` compressed; everything else (models,
embeddings, streaming responses, errors) passes through unchanged.

Routes that are compressed:
- ``POST /v1/chat/completions``  (OpenAI and compatible: Groq, OpenRouter, vLLM, LM Studio, Ollama /v1 ...)
- ``POST /v1/messages``          (Anthropic; the summary goes into ``system``)
- ``POST /api/chat``             (Ollama native)

Each conversation gets its own Compressor, so the prompt-cache-friendly
prefix is kept per conversation. A conversation is identified by the
``X-Delbina-Session`` header if the client sends one, otherwise by model +
system prompt + first message. Only Python's standard library is used.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from . import __version__
from .messages import Compressor, fingerprint, message_text

MAX_BODY = 64 * 1024 * 1024
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
}
SESSION_HEADER = "x-delbina-session"


@dataclass
class Totals:
    requests: int = 0
    compressed_requests: int = 0
    compactions: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, before: int, after: int, compacted: bool) -> None:
        with self.lock:
            self.compressed_requests += 1
            self.compactions += compacted
            self.tokens_before += before
            self.tokens_after += after

    def as_dict(self) -> dict[str, Any]:
        with self.lock:
            saved = 1 - self.tokens_after / self.tokens_before if self.tokens_before else 0.0
            return {
                "requests": self.requests, "compressed_requests": self.compressed_requests,
                "compactions": self.compactions, "tokens_before": self.tokens_before,
                "tokens_after": self.tokens_after, "saved": round(saved, 4),
            }


class Sessions:
    """LRU map of conversation key -> Compressor."""

    def __init__(self, factory: Callable[[dict[str, str]], Compressor], max_sessions: int = 256):
        self.factory = factory
        self.max_sessions = max_sessions
        self._items: OrderedDict[str, Compressor] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, headers: dict[str, str]) -> Compressor:
        with self._lock:
            comp = self._items.get(key)
            if comp is None:
                comp = self._items[key] = self.factory(headers)
                while len(self._items) > self.max_sessions:
                    _, old = self._items.popitem(last=False)
                    old.close()
            self._items.move_to_end(key)
            return comp


def session_key(kind: str, body: dict, messages: list[dict], headers: dict[str, str]) -> str:
    explicit = headers.get(SESSION_HEADER)
    if explicit:
        return "h:" + explicit
    first_user = next((m for m in messages if m.get("role") != "system"), None)
    parts = [kind, str(body.get("model", ""))]
    parts += [fingerprint(m) for m in messages if m.get("role") == "system"]
    parts.append(fingerprint(first_user) if first_user else "")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=12).hexdigest()


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return message_text({"content": system})


def compress_body(kind: str, body: dict, sessions: Sessions, headers: dict[str, str]):
    """Compress a request body in place. Returns the Stats or None."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    if kind == "anthropic":
        system = body.get("system")
        lead = [{"role": "system", "content": _system_text(system)}] if system else []
        full = lead + messages
        comp = sessions.get(session_key(kind, body, full, headers), headers)
        out = comp.compress(full)
        n_sys = 0
        while n_sys < len(out) and out[n_sys].get("role") == "system":
            n_sys += 1
        extra = [m["content"] for m in out[len(lead):n_sys]]  # the summary, if any
        if extra:
            if isinstance(system, list):
                body["system"] = system + [{"type": "text", "text": t} for t in extra]
            else:
                body["system"] = "\n\n".join(([system] if system else []) + extra)
        body["messages"] = out[n_sys:]
    else:
        comp = sessions.get(session_key(kind, body, messages, headers), headers)
        body["messages"] = comp.compress(messages)
    return comp.last_stats


def route_kind(method: str, path: str) -> str | None:
    if method != "POST":
        return None
    p = urllib.parse.urlsplit(path).path.rstrip("/")
    if p.endswith("/chat/completions"):
        return "openai"
    if p.endswith("/v1/messages"):
        return "anthropic"
    if p == "/api/chat":
        return "ollama"
    return None


def upstream_url(upstream: str, path: str) -> str:
    """/v1/... -> <upstream>/... ; anything else -> <upstream origin>/path."""
    base = upstream.rstrip("/")
    if path.startswith("/v1/") or path == "/v1":
        return base + path[3:]
    parts = urllib.parse.urlsplit(base)
    return f"{parts.scheme}://{parts.netloc}{path}"


def make_handler(upstream: str, sessions: Sessions, totals: Totals, log: bool, timeout: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"delbina/{__version__}"

        def log_message(self, fmt, *args):  # quiet default access log
            pass

        def do_GET(self):
            if self.path.rstrip("/") == "/delbina/stats":
                return self._send_json(200, totals.as_dict())
            self._forward(None)

        def do_POST(self):
            self._forward(self._read_body())

        do_PUT = do_PATCH = do_POST
        do_DELETE = do_GET

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            return self.rfile.read(length) if length else b""

        def _send_json(self, status: int, data: dict) -> None:
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _forward(self, raw: bytes | None) -> None:
            with totals.lock:
                totals.requests += 1
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            lower = {k.lower(): v for k, v in headers.items()}
            headers = {k: v for k, v in headers.items() if k.lower() != SESSION_HEADER}
            kind = route_kind(self.command, self.path)
            stats = None
            if kind and raw:
                try:
                    body = json.loads(raw)
                    stats = compress_body(kind, body, sessions, lower)
                    if stats is not None:
                        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                        totals.add(stats.tokens_before, stats.tokens_after, stats.compacted)
                        if log:
                            print(f"[delbina] {body.get('model', '?')}: {stats.tokens_before} -> "
                                  f"{stats.tokens_after} tokens ({stats.saved:.0%} saved)"
                                  f"{', compacted' if stats.compacted else ''}"
                                  f"{f', {stats.trimmed} tool outputs trimmed' if stats.trimmed else ''}",
                                  file=sys.stderr, flush=True)
                except Exception as e:  # never break the client because of compression
                    if log:
                        print(f"[delbina] compression skipped: {e!r}", file=sys.stderr, flush=True)

            req = urllib.request.Request(upstream_url(upstream, self.path), data=raw, method=self.command,
                                         headers={**headers, "Accept-Encoding": "identity"})
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
            except urllib.error.HTTPError as e:
                resp = e
            except Exception as e:
                return self._send_json(502, {"error": {"message": f"delbina: upstream unreachable: {e}",
                                                       "type": "delbina_upstream_error"}})
            with resp:
                self.send_response(resp.status if hasattr(resp, "status") else resp.code)
                for k, v in resp.headers.items():
                    if k.lower() not in HOP_BY_HOP and k.lower() != "content-encoding":
                        self.send_header(k, v)
                if stats is not None:
                    self.send_header("X-Delbina-Tokens", f"{stats.tokens_before}->{stats.tokens_after}")
                status = resp.status if hasattr(resp, "status") else resp.code
                if status in (204, 304) or self.command == "HEAD":
                    self.end_headers()
                    return
                length = resp.headers.get("Content-Length")
                if length is not None and "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
                    data = resp.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                read = getattr(resp, "read1", resp.read)
                while True:
                    chunk = read(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

    return Handler


def serve(
    upstream: str,
    host: str = "127.0.0.1",
    port: int = 8787,
    compressor: Callable[[dict[str, str]], Compressor] | None = None,
    max_sessions: int = 256,
    log: bool = True,
    timeout: float = 600.0,
) -> ThreadingHTTPServer:
    """Create the proxy server (call .serve_forever() on the result).

    ``compressor`` builds a Compressor for each new conversation; it receives
    that conversation's first request headers (lower-cased).
    """
    factory = compressor or (lambda headers: Compressor())
    sessions = Sessions(factory, max_sessions)
    totals = Totals()
    handler = make_handler(upstream, sessions, totals, log, timeout)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.totals = totals  # type: ignore[attr-defined]
    return server
