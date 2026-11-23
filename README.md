<p align="center"><img src="https://raw.githubusercontent.com/delbina/delbina/main/brand/delbina-logo-transparent-1024.png" width="160" alt="delbina"></p>

# delbina

A context-compression layer that sits in front of existing LLMs to cut the token cost of long chats.

Every chat turn re-sends the whole history, so total token cost grows roughly quadratically with chat length. delbina measures that cost and compresses the history before it reaches the model.

> **Status: alpha (0.x).** Token-cost measurement and simple strategies work. Answer-quality evaluation is not built yet, so savings numbers say nothing about whether the model still answers correctly. The API may change.

## Install

```bash
pip install delbina
```

On first use, tiktoken downloads its encoding file, so the first run needs network access.

## Quick start

```python
from delbina import cost, full, last_n, rolling_summary

turns = ["user: ...", "assistant: ...", "user: ...", "assistant: ..."]

print(cost(turns, full))        # tokens if full history is re-sent every turn
print(cost(turns, last_n(4)))   # tokens if only the last 4 turns are sent

context = rolling_summary(keep=2)(turns)   # what you would actually send
```

### With a local model (Ollama)

```python
from delbina import ollama_summarizer, rolling_summary

strategy = rolling_summary(keep=4, summarize=ollama_summarizer("qwen2.5:1.5b"))
```

Requires a running [Ollama](https://ollama.com) server. No extra Python dependencies.

## CLI

```bash
delbina bench chat.json                       # compare strategies
delbina bench chat.json --ollama qwen2.5:1.5b # use a local model for summaries
```

`chat.json` format: `{"turns": ["...", "..."]}`

## Strategies

| Strategy | What is sent each turn |
|---|---|
| `full` | the whole history (baseline) |
| `last_n(n)` | the last n turns |
| `head_tail(h, t)` | the first h turns (task setup) + the last t turns |
| `rolling_summary(keep, summarize)` | a summary of older turns + the last `keep` turns |

Note: `cost()` counts tokens sent to the main model. It does not yet count the tokens spent by the summarizer itself.

## Development

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest -q
```

## License

Apache-2.0
