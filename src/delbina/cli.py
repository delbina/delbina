"""Command line.

    delbina bench chat.json [--ollama MODEL]
    delbina compress chat.json --max-tokens 2000 [--keep-last 6] [--ollama MODEL] [-o out.json]

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
    if "messages" in data:
        messages = data["messages"]
        return [message_text(m) for m in messages], messages
    turns = data["turns"]
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]
    return turns, messages


def _summarizer(model: str | None):
    if not model:
        return None
    from .ollama import ollama_summarizer
    return ollama_summarizer(model)


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


def compress_file(path: Path, max_tokens: int, keep_last: int, ollama_model: str | None, out: Path | None) -> None:
    _, messages = _load(path)
    comp = Compressor(max_tokens=max_tokens, keep_last=keep_last, summarize=_summarizer(ollama_model))
    result = comp.compress(messages)
    text = json.dumps({"messages": result}, indent=2, ensure_ascii=False)
    if out:
        out.write_text(text, encoding="utf-8")
    else:
        print(text)
    s = comp.last_stats
    print(f"{s.tokens_before} -> {s.tokens_after} tokens ({s.saved:.0%} saved), "
          f"{s.dropped} messages dropped{', summarized' if s.summarized else ''}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="delbina", description="Measure and compress chat context token cost.")
    p.add_argument("--version", action="version", version=f"delbina {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="compare strategy token cost on a chat file")
    b.add_argument("file", type=Path)
    b.add_argument("--ollama", metavar="MODEL", help="use a local Ollama model for rolling_summary")

    c = sub.add_parser("compress", help="compress a chat file to a token budget")
    c.add_argument("file", type=Path)
    c.add_argument("--max-tokens", type=int, default=4000)
    c.add_argument("--keep-last", type=int, default=6)
    c.add_argument("--ollama", metavar="MODEL", help="summarize dropped messages with a local Ollama model")
    c.add_argument("-o", "--output", type=Path, help="write result here instead of stdout")

    args = p.parse_args(argv)
    if args.cmd == "bench":
        bench(args.file, args.ollama)
    elif args.cmd == "compress":
        compress_file(args.file, args.max_tokens, args.keep_last, args.ollama, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
