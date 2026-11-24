# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Use any OpenAI-compatible client through the delbina proxy.

Terminal 1 (local Ollama as the real backend):
    ollama pull qwen2.5:1.5b
    python -m delbina serve --upstream http://localhost:11434/v1 --max-tokens 1500

Terminal 2:
    python examples/proxy_client.py

Any other backend: set DELBINA_MODEL (and OPENAI_API_KEY if it needs a key), e.g. Gemini:
    python -m delbina serve --upstream https://generativelanguage.googleapis.com/v1beta/openai --max-tokens 1500
    $env:DELBINA_MODEL = "gemini-3.5-flash-lite"; python examples/proxy_client.py

The client code is ordinary; only the base URL points at delbina. Watch the
proxy log for token counts, and open http://127.0.0.1:8787/delbina/stats.
Uses only the standard library, so no openai package is needed.
"""
import json
import os
import urllib.request

BASE_URL = "http://127.0.0.1:8787/v1"   # instead of http://localhost:11434/v1
MODEL = os.environ.get("DELBINA_MODEL", "qwen2.5:1.5b")
KEY = os.environ.get("OPENAI_API_KEY", "")
HEADERS = {"Content-Type": "application/json", "X-Delbina-Session": "demo",
           **({"Authorization": f"Bearer {KEY}"} if KEY else {})}


def chat(messages: list[dict]) -> str:
    body = json.dumps({"model": MODEL, "messages": messages}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r:
        print("   ", r.headers.get("X-Delbina-Tokens"), "tokens (before->after)")
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> None:
    history = [{"role": "system", "content": "You are a concise assistant."}]
    for q in ["Name three sorting algorithms.", "Which is fastest on average?", "Explain it in two lines.",
              "What is its worst case?", "How do libraries avoid that?", "Summarize our chat in one line."]:
        history.append({"role": "user", "content": q})
        print(f"you> {q}")
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}\n")
    stream(history + [{"role": "user", "content": "Say goodbye in five words."}])


def stream(messages: list[dict]) -> None:
    """Same proxy, streaming reply (server-sent events) printed as it arrives."""
    body = json.dumps({"model": MODEL, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, headers=HEADERS)
    print("stream> ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                choices = json.loads(line[5:]).get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                print(delta.get("content") or "", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
