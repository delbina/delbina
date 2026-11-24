# Security policy

## Reporting

Please do not open public issues for security problems. Use GitHub's private reporting: **Security → Report a vulnerability** on this repository. Expect a first response within 7 days.

## Scope notes

- delbina does not send data anywhere by itself. Network calls happen only through a summarizer you configure (e.g. `ollama_summarizer`, which talks to the Ollama host you pass, `localhost` by default) and tiktoken's one-time tokenizer download.
- `delbina serve` listens on `127.0.0.1` by default and forwards requests, including API keys in headers, only to the `--upstream` you set. Keys are not logged or written to disk; with `--summary-model` the client's key is held in memory to call the same upstream. Listening on another interface lets anyone who can reach the port use your upstream, including key-less ones such as Ollama.
- Summaries are produced by a model and inserted into your prompt. Treat them like any other model output: chat content can carry prompt-injection text into the summary.
- `redact=` / `--redact` hides common secret formats from the summarizer. It is pattern-based and best-effort; do not rely on it as the only control for highly sensitive data. In mask mode the real values stay in process memory and in saved state files.
- Compaction log lines and `CompactionEvent`s contain counts and timings only, never message text or secret values.
- Releases are published from GitHub Actions with PyPI trusted publishing (OIDC); no long-lived upload tokens exist.
