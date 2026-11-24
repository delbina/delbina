# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Terminal chat with a local Ollama model, history kept under a token budget.

    ollama pull qwen2.5:1.5b
    python examples/ollama_chat.py --model qwen2.5:1.5b --max-tokens 1500

Needs only delbina and a running Ollama server. Type 'exit' to quit.
"""
import argparse
import json
import urllib.request

from delbina import Compressor, ollama_summarizer


def ollama_chat(model: str, messages: list[dict], host: str) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--keep-last", type=int, default=6)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--no-summary", action="store_true", help="drop old turns instead of summarizing")
    args = p.parse_args()

    comp = Compressor(
        max_tokens=args.max_tokens,
        keep_last=args.keep_last,
        summarize=None if args.no_summary else ollama_summarizer(args.model, host=args.host),
    )
    history = [{"role": "system", "content": "You are a concise, helpful assistant."}]

    while True:
        user = input("\nyou> ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = ollama_chat(args.model, comp.compress(history), args.host)
        history.append({"role": "assistant", "content": reply})
        s = comp.last_stats
        print(f"\nbot> {reply}")
        print(f"     [{s.tokens_before} -> {s.tokens_after} tokens, {s.saved:.0%} saved]")


if __name__ == "__main__":
    main()
