<p align="center"><img src="https://raw.githubusercontent.com/delbina/delbina/main/brand/delbina-logo-transparent-1024.png" width="160" alt="delbina"></p>

<h1 align="center">delbina</h1>

<p align="center">Cut the token bill of long chats and AI agents. A Python library, or a drop-in proxy: change one URL, keep your code.</p>

<p align="center">
<a href="https://pypi.org/project/delbina/"><img src="https://img.shields.io/pypi/v/delbina" alt="PyPI"></a>
<a href="https://github.com/delbina/delbina/actions/workflows/test.yml"><img src="https://github.com/delbina/delbina/actions/workflows/test.yml/badge.svg" alt="tests"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="license"></a>
</p>

---

## Why

Chat models are stateless. Every call re-sends the **entire** conversation, so turn 50 pays for turns 1 to 49 again. In agents it is worse: old tool outputs (whole files, search results, test logs) are re-sent on every step.

Simply cutting old messages is not enough. Providers give a large discount (up to 90%) on a prompt whose **beginning is identical** to the previous call's: **prompt caching**. A sliding window changes the beginning every turn, loses that discount, and can cost *more* than sending everything.

delbina compresses in a cache-friendly way:

- **Compaction events, not a sliding window.** When the chat overflows the budget, delbina compresses it once to 60% of the budget, then lets it grow again. Between events the prefix stays byte-identical, so it stays cached.
- **Old tool outputs are trimmed first.** Large tool results that the agent already used are shortened to their start and end. The newest ones stay intact.
- **Optional running summary** of dropped messages, from a local or cheap model, prepared in the background so it does not delay replies.

```
agent / app ──► delbina (library call or proxy) ──► model API
 full history       fits budget, stable prefix        cached + cheap
```

## Results

Replay of a 25-task coding-agent session (269 messages, 134 model calls, ~100k tokens of history), with cached input at 0.1x the normal price (Anthropic, OpenAI GPT-4.1/GPT-5, DeepSeek). Reproduce with `delbina cost bench/data/agent_session.json --max-tokens N`.

| budget | method | billed input vs full history | user turns still visible |
|---|---|---|---|
| 32k | full history | 1.00 | 13.0 |
| 32k | sliding window | **1.08** (more expensive!) | 6.5 |
| 32k | **delbina** | **0.57** | **9.0** |
| 64k | sliding window | 1.51 | 11.0 |
| 64k | **delbina** | **0.82** | **12.8** |
| 16k | sliding window | 0.59 | 3.5 |
| 16k | **delbina** | **0.40** | **5.0** |

Token counts in this table use the `chars` estimate; exact numbers shift slightly with a real tokenizer, ratios do not. Answer quality is measured separately with `delbina eval` (see [Quality](#quality)).

**New here?** [docs/getting-started.md](docs/getting-started.md) walks through setup step by step, for free tiers (Gemini, Groq, OpenRouter) and paid APIs (OpenAI, Anthropic, ...).

## Install

```bash
pip install delbina
```

Python 3.10+. The only dependency is `tiktoken`, which downloads its tokenizer file on first use.

## Option 1: proxy, zero code changes

```bash
delbina serve --upstream https://api.openai.com/v1 --max-tokens 16000
```

Then point any OpenAI-compatible client or agent at `http://127.0.0.1:8787/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1")   # the only change
```

The official OpenAI SDKs, and most tools built on them, read `OPENAI_BASE_URL`, so often you only set an environment variable. Your API key passes through untouched and is never stored. Also works with Anthropic (`/v1/messages`), Ollama, vLLM, LM Studio, Groq, OpenRouter and any other OpenAI-compatible server. Each conversation gets its own compressor; live savings are at `http://127.0.0.1:8787/delbina/stats`.

## Option 2: library

```python
from delbina import Compressor

comp = Compressor(max_tokens=16000)   # one per conversation

messages = [{"role": "system", "content": "You are a helpful assistant."}]
while True:
    messages.append({"role": "user", "content": input("> ")})
    reply = call_your_model(comp.compress(messages))   # full history stays in `messages`
    messages.append({"role": "assistant", "content": reply})
    print(comp.last_stats)
```

Guarantees:

- leading **system** messages are always kept
- the newest user request and everything after it, and the last `keep_last` messages, are never dropped
- the kept part always starts on a **user** message, so a tool call is never separated from its result
- tool-call ids and all other message fields are preserved; your input list is never modified
- between compaction events the output only grows at the end (stable prefix for prompt caching)

Add a running summary of dropped messages:

```python
from delbina import Compressor, ollama_summarizer

comp = Compressor(max_tokens=16000, summarize=ollama_summarizer("qwen2.5:1.5b"), background=True)
```

Keep the state across restarts:

```python
comp.load("chat-42.delbina.json")   # no-op if the file doesn't exist
...
comp.save("chat-42.delbina.json")
```

Async apps: `await comp.acompress(messages)` accepts sync or `async` summarizers and never blocks the event loop.

Ready-to-use examples for OpenAI, Anthropic, Ollama and agents: [docs/integrations.md](docs/integrations.md), [examples/](examples/).

## Quality

Savings only count if answers stay correct. `delbina eval` plants a fact (a key, a port, an ID) early, midway or late in long agent conversations, then asks the model for it under each compression mode:

```bash
delbina eval --model qwen2.5:7b --summary-model qwen2.5:1.5b    # local Ollama, free
delbina eval --base-url https://api.openai.com/v1 --model gpt-4o-mini --summary-model gpt-4o-mini
delbina eval --offline                                           # no model: is the fact still in context?
```

No local GPU? The Gemini API free tier works through its OpenAI-compatible endpoint. Create a key in Google AI Studio, copy a current Flash or Flash-Lite model ID from there, and keep within its rate limits:

```powershell
$env:GEMINI_API_KEY = "your-key"
python -m delbina eval --base-url https://generativelanguage.googleapis.com/v1beta/openai `
  --model MODEL_ID --summary-model MODEL_ID --reasoning-effort minimal --rpm 8 --cases 12
```

`eval` retries rate-limit errors, and `--modes full,sliding` / `--modes delbina,summary` split a run across days if the daily quota is small.

The honest result so far: drop-only compression forgets facts that fall out of the window. With a budget of 3000 tokens, 0% of early facts survive in context. The summary mode exists for exactly this, and `eval` measures how much it recovers with your models. Set the budget and summarizer for your workload with this, not with guesses.

## CLI

```bash
delbina serve --upstream URL [--max-tokens 16000] [--summary-model gpt-4o-mini | --ollama MODEL] [--background]
delbina cost chat.json --max-tokens 16000 [--cache-price 0.1]   # billed-token estimate with prompt caching
delbina eval [--model M --summary-model M | --offline]           # answer-quality benchmark
delbina compress chat.json --max-tokens 4000 -o small.json
delbina --version
```

`chat.json` is `{"messages": [...]}`, a bare list of messages, or `{"turns": ["...", ...]}`. On Windows, if `delbina.exe` is blocked by Smart App Control, use `python -m delbina ...`.

## Token counting

Default counting uses tiktoken `cl100k_base`. Pass a model name for better counts:

```python
Compressor(count_tokens="gpt-4o")                        # exact (o200k_base)
Compressor(count_tokens="hf:Qwen/Qwen2.5-7B-Instruct")   # exact, pip install "delbina[hf]"
Compressor(count_tokens="claude-sonnet-4")               # estimate: tokenizer not public
Compressor(count_tokens="chars")                         # no tokenizer, no network
```

## Status

**Alpha (0.x).** The API may change between minor versions. See [CHANGELOG.md](CHANGELOG.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md).

## License

Copyright 2026 Krupakar S Bilichod. Licensed under [Apache-2.0](LICENSE).
If you redistribute delbina or a modified version, keep the [NOTICE](NOTICE) file with it.
