# delbina guide

## Concepts

A chat model only sees what you send it, and chat apps re-send the whole history on every call:

- **cost** grows with every turn (turn *n* pays for turns 1..n),
- agents re-send old **tool outputs** (files, search results, logs), the largest part of their context,
- long chats eventually exceed the model's **context window**.

### Prompt caching

OpenAI, Anthropic, DeepSeek, Gemini and local servers (vLLM, llama.cpp) cache the beginning of a prompt. If a call starts with exactly the same tokens as an earlier one (at least ~1024 tokens), that part is billed at a discount (0.1x on Anthropic and recent OpenAI models, 0.5x on older OpenAI models) and processed faster.

This changes what good compression looks like. A **sliding window** that drops one old message per turn changes the beginning of every prompt, so nothing is ever cached. It sends fewer tokens but can bill more than sending everything.

delbina works in **compaction events** instead:

```
tokens
 max ┤        /|        /|        /|
     │       / |       / |       / |
 60% ┤      /  |/     /  |/     /  |/
     │     /         /         /
     └──────────────────────────────── turns
           ↑ event   ↑ event   ↑ event   (prefix changes only here)
```

Between events, new messages are appended and everything before them is unchanged, so it is cached. At an event, delbina:

1. shortens old tool outputs (all but the newest `keep_tool_results`) to about `tool_max_tokens`,
2. if that is not enough, drops (and optionally summarizes) the oldest messages,
3. until the chat is at `target_ratio * max_tokens`.

Lower `target_ratio` means fewer events (better caching) but less context after each event. 0.5 to 0.7 is a good range; 1.0 gives the old sliding-window behaviour.

## `Compressor`

Use **one per conversation**. It keeps the state (cut point, trimmed tool outputs, summary) that makes the prefix stable.

```python
Compressor(
    max_tokens=4000,        # budget for the returned list
    keep_last=6,            # last N messages are never dropped
    summarize=None,         # list[str] -> str, sync or async; None = drop only
    count_tokens=None,      # str -> int, or a model name ("gpt-4o", "hf:<repo>", "chars"); default cl100k_base
    summary_role="system",  # role of the inserted summary message
    summary_tokens=300,     # budget reserved for the summary when summarize is set
    align_to_user=True,     # kept part starts on a user message (never splits tool call/result)
    target_ratio=0.6,       # compress to this fraction of max_tokens on overflow
    tool_max_tokens=400,    # old tool outputs shortened to ~N tokens; None = off
    keep_tool_results=2,    # newest N tool outputs are never shortened
    background=False,       # prepare the next summary in a background thread
    prefetch_ratio=0.85,    # ... starting at this fraction of max_tokens
)
```

| Method / attribute | Purpose |
|---|---|
| `compress(messages)` | return a list that fits the budget |
| `await acompress(messages)` | same, for async code; works with async summarizers |
| `last_stats` | `Stats(tokens_before, tokens_after, dropped, summarized, trimmed, compacted)`, plus `.saved` |
| `summary` | current running summary |
| `state()` / `load_state(d)` | JSON-serializable state |
| `save(path)` / `load(path)` | the same, to a file; `load` returns False if the file is missing |
| `reset()` | forget everything (new conversation) |
| `close()` | stop the background thread |
| `tokens(messages)` | approximate token count of a list |

`last_stats.compacted` is True when the call changed the prefix (expect a cache miss on that call).

### What is always kept

- leading `system` messages,
- the newest user message and everything after it (the current request and its tool calls),
- the last `keep_last` messages.

These are never dropped, but old tool outputs among them can still be shortened. If they alone exceed `max_tokens`, the result is over budget; check `last_stats.tokens_after`.

### Safety rules

- The kept part starts on a user message that is not a tool result, so every kept tool result still has its tool call.
- Tool outputs are shortened in place: `tool_call_id`, `tool_use_id`, `name` and every other key are preserved. A marker `[... delbina trimmed N tokens of tool output ...]` shows the model what happened.
- In Anthropic format, only `tool_result` blocks are shortened, never the user's own text blocks.
- If the history is edited (an earlier message changes, the system prompt changes), the saved state no longer matches and is discarded automatically.

### Summaries

With `summarize`, dropped messages are replaced by one message `"Summary of the earlier conversation:\n..."`. At each event the summarizer receives only:

```
["previous summary: <old summary>", "user: <newly dropped>", "tool: <shortened output>", ...]
```

so summarization cost stays flat as the chat grows, and runs once per event instead of once per turn.

With `background=True`, delbina starts summarizing in a background thread once the chat passes `prefetch_ratio * max_tokens`. When the event comes, the summary is usually ready and adds no delay.

Summarizers: `ollama_summarizer(model)` (local Ollama) and `openai_summarizer(model, base_url=..., api_key=...)` (any OpenAI-compatible API), both standard-library only. Or any function `list[str] -> str`.

### Message format

Each message is a dict with `role` and `content`. `content` can be a string or a list of blocks. Text, `tool_result` and `tool_use` blocks and OpenAI `tool_calls` are all counted. Tool results are recognised as `role: "tool"`, `role: "function"`, or Anthropic `tool_result` blocks.

### `compress(messages, **options)`

One-shot helper. A single call has no memory of earlier turns, so it cannot keep the prefix stable; it fits the chat just under `max_tokens` (`target_ratio=1.0` by default). Use a `Compressor` in loops.

## Proxy (`delbina serve`)

```bash
delbina serve --upstream https://api.openai.com/v1 --max-tokens 16000
```

| Option | Default | Meaning |
|---|---|---|
| `--upstream URL` | required | the real API base URL |
| `--host`, `--port` | 127.0.0.1, 8787 | where to listen |
| `--max-tokens`, `--keep-last`, `--target`, `--tool-max-tokens`, `--keep-tool-results`, `--tokenizer` | 16000, 6, 0.6, 400, 2, cl100k | as in `Compressor` |
| `--summary-model M` | off | summarize through the upstream with model M, using the client's own API key |
| `--ollama M` | off | summarize with a local Ollama model |
| `--background` | off | prepare summaries ahead of time |
| `--max-sessions` | 256 | conversations kept in memory (least recently used are evicted) |
| `--quiet` | off | no per-request log line |

Routes: `POST .../chat/completions` (OpenAI-compatible), `POST /v1/messages` (Anthropic: the summary is appended to `system`), `POST /api/chat` (Ollama native). Everything else, including streaming responses and error statuses, is forwarded unchanged. Requests whose body is not valid JSON are forwarded unmodified.

Path mapping: `/v1/<rest>` goes to `<upstream>/<rest>`, any other path to the upstream's origin. So with `--upstream https://openrouter.ai/api/v1`, clients still use `http://127.0.0.1:8787/v1`.

Conversations are told apart by the `X-Delbina-Session` header if present (recommended for agents running many similar chats), otherwise by model + system prompt + first message. The header is not forwarded upstream.

`GET /delbina/stats` returns totals: requests, compactions, tokens before/after, saved fraction. Responses carry an `X-Delbina-Tokens: before->after` header.

Security: the proxy listens on 127.0.0.1 by default. API keys pass through in headers and are held in memory only for `--summary-model`; nothing is written to disk. Listening on other interfaces prints a warning.

## Measuring

| Command / function | What it tells you |
|---|---|
| `delbina cost FILE --max-tokens N` / `delbina.costsim.compare` | billed input tokens with prompt caching: full history vs sliding window vs delbina, plus how much context survives |
| `delbina eval` / `delbina.evaluate` | answer quality: can the model still recall planted facts after compression? |
| `delbina bench FILE` | simple strategies over plain lists of turns |
| `count`, `counter(model)`, `cumulative_cost`, `cost` | token measurement helpers |

`delbina cost` assumes a model call after every user and tool message and treats the identical leading messages of consecutive calls as cached (if at least `--min-prefix` tokens). It ignores Anthropic's 1.25x cache-write surcharge, which slightly favours the full-history baseline.

## Token counting

| `count_tokens=` | Counting |
|---|---|
| `None` | tiktoken `cl100k_base` |
| `"gpt-4o"`, `"gpt-4.1"`, `"gpt-5"`, `"o3"` ... | exact, `o200k_base` |
| `"gpt-4"`, `"gpt-3.5-turbo"` | exact, `cl100k_base` |
| `"hf:<repo>"` | exact, Hugging Face tokenizer (`pip install "delbina[hf]"`, downloads the tokenizer once) |
| `"tiktoken:<encoding>"` | any tiktoken encoding |
| `"claude..."`, `"llama..."`, `"qwen..."`, `"mistral..."` ... | estimate from cl100k with a per-family ratio |
| `"chars"` | `len(text) / 4`, no dependencies, no network |
| any `str -> int` function | yours |
