<p align="center"><img src="https://raw.githubusercontent.com/delbina/delbina/main/brand/delbina-logo-transparent-1024.png" width="160" alt="delbina"></p>

<h1 align="center">delbina</h1>

<p align="center">Keep long LLM chats under a token budget. One function call, no server, no lock-in.</p>

<p align="center">
<a href="https://pypi.org/project/delbina/"><img src="https://img.shields.io/pypi/v/delbina" alt="PyPI"></a>
<a href="https://github.com/delbina/delbina/actions/workflows/test.yml"><img src="https://github.com/delbina/delbina/actions/workflows/test.yml/badge.svg" alt="tests"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="license"></a>
</p>

---

## Why

Chat models are stateless. Every turn, your app re-sends the **entire** conversation, so a 50-turn chat pays for turn 1 fifty times. Total cost grows roughly with the square of chat length, and eventually the chat hits the context limit.

delbina sits between your chat history and the model call. It trims the history to a token budget you choose, keeps what matters (system prompt, recent turns), and can replace older turns with a running summary from a local model.

```
your messages ──► delbina.compress() ──► model API
  (full history)      (fits budget)
```

## Install

```bash
pip install delbina
```

Python 3.10+. The only dependency is `tiktoken` (for counting tokens). On first use it downloads a tokenizer file, so the first run needs network access.

## Quick start

```python
from delbina import compress

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # ... hundreds more turns
]

small = compress(messages, max_tokens=2000)
# send `small` to your model instead of `messages`
```

What `compress` guarantees:

- leading **system** messages are always kept
- the last `keep_last` messages (default 6) are always kept
- older messages are dropped, oldest first, until the chat fits `max_tokens`
- the kept part starts on a **user** message, so tool calls and replies are never split
- your input list is never modified

## In a chat loop (recommended)

Use one `Compressor` per conversation. With a summarizer, dropped turns become a summary, and the summary is **updated incrementally**: each turn only the newly dropped messages are summarized, not the whole history again.

```python
from delbina import Compressor, ollama_summarizer

comp = Compressor(
    max_tokens=3000,
    keep_last=6,
    summarize=ollama_summarizer("qwen2.5:1.5b"),  # any local Ollama model
)

messages = [{"role": "system", "content": "You are a helpful assistant."}]
while True:
    messages.append({"role": "user", "content": input("> ")})
    reply = call_your_model(comp.compress(messages))   # full history stays in `messages`
    messages.append({"role": "assistant", "content": reply})
    print(reply)
    print(comp.last_stats)   # tokens_before, tokens_after, dropped, summarized
```

The summarizer is any function `list[str] -> str`, so you can plug in any model or API.

Ready-to-use examples for **OpenAI**, **Anthropic** and **Ollama**: [docs/integrations.md](docs/integrations.md) and [examples/](examples/).

## CLI

```bash
delbina compress chat.json --max-tokens 2000 -o small.json   # compress a saved chat
delbina bench chat.json                                       # compare strategies' token cost
delbina --version
```

`chat.json` is `{"messages": [{"role": ..., "content": ...}, ...]}` or `{"turns": ["...", ...]}`.

On Windows, if `delbina.exe` is blocked by Smart App Control, use `python -m delbina ...` instead.

## Measuring

```python
from delbina import cost, full, last_n, rolling_summary

turns = ["...", "...", "..."]
cost(turns, full)        # tokens spent if the full history is re-sent every turn
cost(turns, last_n(4))   # tokens spent if only the last 4 turns are sent
```

See [docs/guide.md](docs/guide.md) for every option and strategy.

## Status

**Alpha (0.x).** The API may change between minor versions.

What works: token budgeting, system/tail preservation, safe cut points, incremental summaries, measurement tools.

Not built yet: **answer-quality evaluation.** Token savings alone don't prove the model still answers well; a benchmark for that is the next milestone. Token counts use OpenAI's `cl100k_base` encoding, which is an approximation for other models; pass `count_tokens=` for exact counts.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE)
