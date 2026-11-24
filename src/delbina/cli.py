# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Command line.

    delbina serve --upstream URL [--max-tokens N] [--port 8787] ...
    delbina compress chat.json --max-tokens 2000 [-o out.json]
    delbina cost chat.json --max-tokens 16000 [--cache-price 0.1]
    delbina eval [--model MODEL --summary-model MODEL | --offline]
    delbina bench chat.json [--ollama MODEL]

chat.json is either {"turns": ["...", ...]} or {"messages": [{"role": ..., "content": ...}, ...]}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .messages import Compressor, message_text
from .strategies import cost, full, head_tail, last_n, rolling_summary
from .tokens import count


def _load(path: Path) -> tuple[list[str], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"messages": data}
    if "messages" in data:
        messages = data["messages"]
        return [message_text(m) for m in messages], messages
    turns = data["turns"]
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]
    return turns, messages


def _summarizer(model: str | None, host: str = "http://localhost:11434"):
    if not model:
        return None
    from .ollama import ollama_summarizer
    return ollama_summarizer(model, host=host)


def _add_compression_args(p: argparse.ArgumentParser, max_tokens: int) -> None:
    p.add_argument("--max-tokens", type=int, default=max_tokens, help=f"token budget (default {max_tokens})")
    p.add_argument("--keep-last", type=int, default=6, help="last N messages are never dropped (default 6)")
    p.add_argument("--target", type=float, default=0.6,
                   help="on overflow compress to this fraction of max-tokens; 1.0 disables cache-friendly mode (default 0.6)")
    p.add_argument("--tool-max-tokens", type=int, default=400, help="shorten old tool outputs to ~N tokens; 0 = off (default 400)")
    p.add_argument("--keep-tool-results", type=int, default=2, help="newest N tool outputs are never shortened (default 2)")
    p.add_argument("--tokenizer", metavar="MODEL", help="count tokens for this model, e.g. gpt-4o, claude, hf:Qwen/Qwen2.5-7B-Instruct, chars")
    p.add_argument("--ollama", metavar="MODEL", help="summarize dropped messages with a local Ollama model")
    p.add_argument("--ollama-host", default="http://localhost:11434")


def _compressor_kwargs(a: argparse.Namespace) -> dict:
    return dict(
        max_tokens=a.max_tokens, keep_last=a.keep_last, target_ratio=a.target,
        tool_max_tokens=a.tool_max_tokens or None, keep_tool_results=a.keep_tool_results,
        count_tokens=a.tokenizer,
    )


def bench(path: Path, ollama_model: str | None = None) -> None:
    turns, _ = _load(path)
    summarizer = _summarizer(ollama_model)
    strategies = [full, last_n(4), head_tail(1, 3)]
    strategies.append(rolling_summary(keep=2, summarize=summarizer) if summarizer else rolling_summary(keep=2))

    base = cost(turns, full)
    print(f"{path.name}: {len(turns)} turns, {sum(count(t) for t in turns)} tokens once\n")
    print(f"{'strategy':<16}{'tokens':>8}{'saved':>8}")
    for s in strategies:
        c = cost(turns, s)
        print(f"{s.__name__:<16}{c:>8}{1 - c / base:>8.0%}")


def cost_report(a: argparse.Namespace) -> None:
    from .costsim import compare
    _, messages = _load(a.file)
    runs = compare(messages, a.max_tokens, a.keep_last, a.tokenizer, a.cache_price, a.min_prefix,
                   summarize=_summarizer(a.ollama, a.ollama_host), target_ratio=a.target,
                   tool_max_tokens=a.tool_max_tokens or None, keep_tool_results=a.keep_tool_results)
    base = runs[0].billed or 1
    print(f"{a.file.name}: {len(messages)} messages, {runs[0].calls} model calls, budget {a.max_tokens} tokens, "
          f"cached input priced at {a.cache_price}x\n")
    print(f"{'':<24}{'input':>11}{'cached':>8}{'billed':>11}{'vs full':>9}{'compactions':>13}{'turns kept':>12}")
    for r in runs:
        hit = r.cached_tokens / r.input_tokens if r.input_tokens else 0
        print(f"{r.name:<24}{r.input_tokens:>11,}{hit:>8.0%}{r.billed:>11,.0f}{r.billed / base:>9.2f}"
              f"{r.compactions:>13}{r.avg_turns_kept:>12.1f}")
    print("\nbilled = uncached + cached x cache-price. turns kept = user turns the model still sees per call (average).")


def compress_file(a: argparse.Namespace) -> None:
    _, messages = _load(a.file)
    comp = Compressor(**_compressor_kwargs(a), summarize=_summarizer(a.ollama, a.ollama_host))
    result = comp.compress(messages)
    text = json.dumps({"messages": result}, indent=2, ensure_ascii=False)
    if a.output:
        a.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    s = comp.last_stats
    print(f"{s.tokens_before} -> {s.tokens_after} tokens ({s.saved:.0%} saved), {s.dropped} messages dropped, "
          f"{s.trimmed} tool outputs trimmed{', summarized' if s.summarized else ''}", file=sys.stderr)


def run_server(a: argparse.Namespace) -> None:
    from .ollama import openai_summarizer
    from .proxy import serve

    kwargs = _compressor_kwargs(a)
    ollama = _summarizer(a.ollama, a.ollama_host)

    def factory(headers: dict[str, str]) -> Compressor:
        summarize = ollama
        if a.summary_model:
            # Summarize through the same upstream with the client's own key (kept in memory only).
            auth = headers.get("authorization", "")
            key = auth[7:] if auth.lower().startswith("bearer ") else headers.get("x-api-key", "")
            summarize = openai_summarizer(a.summary_model, base_url=a.upstream, api_key=key)
        return Compressor(**kwargs, summarize=summarize, background=a.background and summarize is not None)

    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {a.host}: anyone who can reach this port can use your upstream "
              "(and any key-less upstream such as Ollama). Keep it on 127.0.0.1 unless you mean it.",
              file=sys.stderr)
    server = serve(a.upstream, a.host, a.port, factory, a.max_sessions, log=not a.quiet)
    print(f"delbina {__version__} proxy on http://{a.host}:{a.port}/v1 -> {a.upstream}\n"
          f"budget {a.max_tokens} tokens, target {a.target:.0%}, stats at http://{a.host}:{a.port}/delbina/stats\n"
          "Ctrl+C to stop.", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="delbina", description="Compress LLM chat context: library, CLI and proxy.")
    p.add_argument("--version", action="version", version=f"delbina {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run an OpenAI-compatible compression proxy")
    s.add_argument("--upstream", required=True, help="real API base URL, e.g. https://api.openai.com/v1 or http://localhost:11434/v1")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--max-sessions", type=int, default=256, help="conversations kept in memory (LRU)")
    s.add_argument("--summary-model", metavar="MODEL",
                   help="summarize dropped messages with this model through the upstream (uses the client's API key)")
    s.add_argument("--background", action="store_true", help="prepare summaries ahead of time (no added latency)")
    s.add_argument("--quiet", action="store_true", help="no per-request log lines")
    _add_compression_args(s, 16000)

    c = sub.add_parser("compress", help="compress a chat file to a token budget")
    c.add_argument("file", type=Path)
    c.add_argument("-o", "--output", type=Path, help="write result here instead of stdout")
    _add_compression_args(c, 4000)

    k = sub.add_parser("cost", help="replay a chat and estimate billed tokens with prompt caching")
    k.add_argument("file", type=Path)
    k.add_argument("--cache-price", type=float, default=0.1, help="cached input price vs normal (default 0.1)")
    k.add_argument("--min-prefix", type=int, default=1024, help="minimum cacheable prefix in tokens (default 1024)")
    _add_compression_args(k, 16000)

    e = sub.add_parser("eval", help="answer-quality benchmark: recall of planted facts after compression")
    e.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible API (default: local Ollama)")
    e.add_argument("--model", default="qwen2.5:7b", help="model that answers the questions")
    e.add_argument("--summary-model", help="also test the summary mode, summarizing with this model")
    e.add_argument("--api-key", help="API key (default: OPENAI_API_KEY env var; not needed for Ollama)")
    e.add_argument("--max-tokens", type=int, default=3000)
    e.add_argument("--cases", type=int, default=30)
    e.add_argument("--steps", type=int, default=40, help="agent steps per conversation")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--tokenizer", metavar="MODEL")
    e.add_argument("--offline", action="store_true", help="no model: check whether the fact survives in the context")
    e.add_argument("--modes", help="comma-separated subset of full,sliding,delbina,summary (default: all)")
    e.add_argument("--rpm", type=float, default=0, help="max requests per minute, e.g. 12 for a free tier (default: no limit)")
    e.add_argument("--retries", type=int, default=5, help="retries on 429/5xx/network errors (default 5)")
    e.add_argument("--timeout", type=float, default=60, help="seconds per request before retrying (default 60)")
    e.add_argument("--reasoning-effort", help='sent as reasoning_effort, e.g. "minimal" for Gemini')

    b = sub.add_parser("bench", help="compare simple strategies' token cost on a chat file")
    b.add_argument("file", type=Path)
    b.add_argument("--ollama", metavar="MODEL", help="use a local Ollama model for rolling_summary")

    a = p.parse_args(argv)
    if a.cmd == "bench":
        bench(a.file, a.ollama)
    elif a.cmd == "compress":
        compress_file(a)
    elif a.cmd == "cost":
        cost_report(a)
    elif a.cmd == "eval":
        from .evaluate import main_eval
        main_eval(a)
    elif a.cmd == "serve":
        run_server(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
