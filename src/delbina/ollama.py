"""Summarizer backed by a local Ollama server (stdlib only, no extra deps)."""
import json
import urllib.request

from .strategies import Summarizer

PROMPT = (
    "Summarize the following chat turns as compactly as possible. "
    "Keep facts, decisions, names, numbers and open questions. "
    "Drop greetings and filler.\n\n{chat}\n\nSummary:"
)


def ollama_summarizer(
    model: str = "qwen2.5:1.5b",
    host: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> Summarizer:
    """Return a summarizer that calls Ollama's /api/generate endpoint."""
    url = host.rstrip("/") + "/api/generate"

    def summarize(turns: list[str]) -> str:
        body = json.dumps({
            "model": model,
            "prompt": PROMPT.format(chat="\n".join(f"- {t}" for t in turns)),
            "stream": False,
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"].strip()

    return summarize
