# Changelog

All notable changes. Versions follow [Semantic Versioning](https://semver.org/); 0.x releases may change the API.

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
