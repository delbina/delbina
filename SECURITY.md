# Security policy

## Reporting

Please do not open public issues for security problems. Use GitHub's private reporting: **Security → Report a vulnerability** on this repository. Expect a first response within 7 days.

## Scope notes

- delbina does not send data anywhere by itself. Network calls happen only through a summarizer you configure (e.g. `ollama_summarizer`, which talks to the Ollama host you pass, `localhost` by default) and tiktoken's one-time tokenizer download.
- Summaries are produced by a model and inserted into your prompt. Treat them like any other model output: chat content can carry prompt-injection text into the summary.
- Releases are published from GitHub Actions with PyPI trusted publishing (OIDC); no long-lived upload tokens exist.
