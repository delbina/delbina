# Changelog

All notable changes. Versions follow [Semantic Versioning](https://semver.org/); 0.x releases may change the API.

## [0.3.0] - 2026-11-23

Focus: make delbina a real middleman for AI agents that lowers the bill, not just the token count.

### Added
- **Prompt-cache-friendly compression.** `Compressor` now works in compaction events: on overflow it compresses once to `target_ratio` (default 0.6) of `max_tokens`, then lets the chat grow with an unchanged prefix, so provider prompt caching keeps hitting. `Stats.compacted` tells when the prefix changed.
- **Tool-output trimming.** Old tool results (OpenAI `tool`/`function` messages, Anthropic `tool_result` blocks) are shortened to their start and end (`tool_max_tokens`, default 400); the newest `keep_tool_results` (default 2) stay intact. IDs and other fields are preserved. `Stats.trimmed` counts them.
- **`delbina serve`**: OpenAI-compatible proxy. Change only `base_url`; supports `/v1/chat/completions`, Anthropic `/v1/messages`, Ollama `/api/chat`, streaming, per-conversation sessions (`X-Delbina-Session`), `/delbina/stats`. Standard library only.
- **`delbina cost`** and `delbina.costsim`: replay a chat and estimate billed input tokens with prompt caching (full history vs sliding window vs delbina).
- **`delbina eval`** and `delbina.evaluate`: answer-quality benchmark with planted facts; `--offline` mode needs no model.
- `counter(model)` and `count_tokens="<model>"`: o200k/cl100k for OpenAI models, Hugging Face tokenizers via `hf:<repo>` (extra `delbina[hf]`), per-family estimates for Claude, Llama, Qwen, Mistral, and `chars`.
- State persistence: `Compressor.state()`, `load_state()`, `save(path)`, `load(path)`. State is discarded automatically if the history no longer matches.
- Async and background summaries: `await Compressor.acompress()` (sync or async summarizers), `background=True` prepares the next summary before the budget is hit.
- `openai_summarizer` for any OpenAI-compatible API (with `extra=` body fields such as `reasoning_effort`).
- `eval` and the summarizers retry 429/5xx with backoff (honouring `Retry-After`); `eval --rpm`, `--reasoning-effort`, `--answer-max-tokens` and `--modes` make free tiers such as Gemini's usable.
- `message_text` now includes tool results, `tool_use` blocks and `tool_calls` arguments, so agent messages are counted correctly.
- `docs/getting-started.md`: step-by-step setup for free-tier and paid APIs.
- Benchmark data `bench/data/agent_session.json` (synthetic coding-agent session) and its generator; examples `agent_loop.py`, `proxy_client.py`.

### Changed
- The newest user request and everything after it are never dropped (before, `keep_last=0` could drop the whole conversation).
- The kept part never starts on a tool result; if no safe cut exists, less is dropped instead.
- `compress()` (one-shot) keeps the 0.2 behaviour of fitting just under `max_tokens` (`target_ratio=1.0`), since a single call cannot keep a stable prefix. Use `Compressor` in loops.
- Summarizers receive shortened tool outputs; the summary is updated once per compaction event instead of every turn.
- Summaries are capped to `summary_tokens` (the room the planner reserves), so a long-winded summarizer can no longer push the result over budget. Built-in summarizers ask for a word limit and default to `max_tokens=300`; `ollama_summarizer` gained `max_tokens`.
- `eval` shows an "in context" column (was the fact still in what the model saw), stops a mode after 2 errors in a row, exits early when no API key is set for a remote API, and includes the server's error message in failures.
- Copyright notice: `NOTICE` file (shipped in the package) and SPDX headers in source files.

## [0.2.0] - 2026-11-23

### Added
- `compress(messages, ...)` and `Compressor`: keep chat message lists (`role`/`content`) under a token budget.
  - Always keeps leading system messages and the last `keep_last` messages.
  - Cuts on user-message boundaries so tool calls and results stay together.
  - Optional summarizer for dropped messages, updated incrementally and cached per conversation.
  - `last_stats` with tokens before/after, dropped count, saved fraction.
- `delbina compress` CLI command; `bench` now also reads `{"messages": [...]}` files.
- `python -m delbina` entry point (works where `delbina.exe` is blocked on Windows).
- `py.typed` marker, docs (`docs/`), examples (`examples/`), CONTRIBUTING, SECURITY.

### Changed
- Module `delbina.compress` renamed to `delbina.strategies` (the name `compress` is now the main function). Top-level imports (`from delbina import cost, last_n, ...`) are unchanged.

## [0.1.0] - 2026-11-23

### Added
- Token counting (`count`, `cumulative_cost`) and strategy cost measurement (`cost`).
- Strategies: `full`, `last_n`, `head_tail`, `rolling_summary`.
- `ollama_summarizer` for local summaries via Ollama.
- `delbina bench` CLI.
