"""Command line: `delbina bench chat.json [--ollama MODEL]`."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compress import cost, full, head_tail, last_n, rolling_summary
from .tokens import count


def bench(path: Path, ollama_model: str | None = None) -> None:
    turns = json.loads(path.read_text(encoding="utf-8"))["turns"]
    summarizer = None
    if ollama_model:
        from .ollama import ollama_summarizer
        summarizer = ollama_summarizer(ollama_model)
    strategies = [full, last_n(4), head_tail(1, 3)]
    strategies.append(rolling_summary(keep=2, summarize=summarizer) if summarizer else rolling_summary(keep=2))

    base = cost(turns, full)
    print(f"{path.name}: {len(turns)} turns, {sum(count(t) for t in turns)} tokens once\n")
    print(f"{'strategy':<16}{'tokens':>8}{'saved':>8}")
    for s in strategies:
        c = cost(turns, s)
        print(f"{s.__name__:<16}{c:>8}{1 - c / base:>8.0%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="delbina", description="Measure and compress chat context token cost.")
    p.add_argument("--version", action="version", version=f"delbina {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="compare strategy token cost on a chat JSON file")
    b.add_argument("file", type=Path, help='JSON file: {"turns": ["...", ...]}')
    b.add_argument("--ollama", metavar="MODEL", help="use a local Ollama model for rolling_summary")
    args = p.parse_args(argv)
    if args.cmd == "bench":
        bench(args.file, args.ollama)
    return 0


if __name__ == "__main__":
    sys.exit(main())
