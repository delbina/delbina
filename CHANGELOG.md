# Changelog

All notable changes. Versions follow [Semantic Versioning](https://semver.org/); 0.x releases may change the API.

## [0.5.0] - 2026-11-28

### Added
- **delbina-sum support**: `DELBINA_SUM` (`hf.co/delbina/delbina-sum-GGUF:Q4_K_M`), a 1.5B summarizer fine-tuned for delbina's prompt and the `Facts:` / `Context:` format that `checked` verifies. Model `delbina/delbina-sum@ff16f9a`, GGUF `delbina/delbina-sum-GGUF@659a80f`; both Q4_K_M and Q8_0 pass every release bar (see the model cards).
- `openai_summarizer(temperature=)`: sent only when set (use 0 for delbina-sum behind vLLM or llama.cpp).

### Changed
- `ollama_summarizer` now sends `temperature=0` and `num_ctx=8192` by default (new parameters). Ollama's defaults sampled at 0.8 and could silently cut long chats; pass `temperature=` / `num_ctx=` to change them.

## [0.4.1] - 2026-11-25

### Added
- **Fact check for summaries**: `check_facts(summary, source)` drops fact lines (`- subject: label: value`, before a `Context:` line) whose value does not occur in the source text, so a summary can lose a fact but never gain one. `checked(summarizer, on_drop=)` wraps any summarizer with it. Returns `FactCheck(text, dropped)`.

### Changed
- CI tests Python 3.10 to 3.15; releases go to TestPyPI before PyPI.

## [0.4.0] - 2026-11-24

Focus: control what is kept, keep secrets away from summarizers, and see what each compaction did.

### Added
- **`protect=`**: messages that are never dropped, summarized or trimmed. Presets `"first_user"` and `"system"`, or a function `(message, index) -> bool`. Protected tool calls keep their results (and the other way round). Pinned messages sit after the summary, so the prefix stays cache-stable. `BudgetError` when they alone exceed `max_tokens`. `Stats.pinned`. CLI: `--protect-task`.
- **Secret redaction** for summarizer input: `redact=True`/`"mask"` (placeholders, restored in the summary) or `"strip"` (one-way). New module `delbina.redact` with `Redactor` and `redact_text`. CLI: `--redact mask|strip`.
- **Compaction log**: `CompactionEvent` (tokens, dropped, pinned, trimmed, summary size, summarizer time, prefetch, redaction counts; never text) on the `"delbina"` logger and via `on_compact=`. `Compressor.last_event`. `delbina serve` prints events unless `--quiet`.

### Changed
- **`strict_budget=True` (default)**: when a compaction would still exceed `max_tokens`, `keep_last` gives way first (the newest user turn is still kept), then the newest tool outputs are shortened to `tool_max_tokens`. Before, `keep_last` plus one large tool output could leave the result well over budget (49 of 252 calls in the bundled agent session at a 3000-token budget; now 0, with fewer compactions). `strict_budget=False` restores the old behaviour.
- `Stats.dropped` no longer counts protected messages.

## [0.3.0] - 2026-11-24

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
- The room reserved for the summary grows to the real size of the current summary, so long summaries no longer push the result over budget; a hard cap at 2x `summary_tokens` catches summarizers that ramble. `ollama_summarizer` gained `max_tokens` (default 400).
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
