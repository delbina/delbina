# delbina guide

## Concepts

A chat model only sees what you send it. Chat apps re-send the whole history on every turn, so:

- **cost** grows with every turn (turn *n* pays for turns 1..n), and
- long chats eventually exceed the model's **context window**.

delbina reduces the history you send, under a **token budget**, while keeping the parts that matter most.

## `compress(messages, **options)`

One-shot compression. Returns a new list; the input is not modified. Accepts every option of `Compressor`.

```python
from delbina import compress
small = compress(messages, max_tokens=2000, keep_last=4)
```

## `Compressor`

```python
Compressor(
    max_tokens=4000,      # budget for the whole returned list
    keep_last=6,          # last N messages are always kept
    summarize=None,       # function list[str] -> str; None = just drop old messages
    count_tokens=None,    # function str -> int; default: tiktoken cl100k_base
    summary_role="system",# role of the inserted summary message
    summary_tokens=300,   # budget reserved for the summary when summarize is set
    align_to_user=True,   # kept part starts on a user message
)
```

| Method / attribute | Purpose |
|---|---|
| `compress(messages)` | return a list that fits the budget |
| `tokens(messages)` | approximate token count of a list |
| `last_stats` | `Stats(tokens_before, tokens_after, dropped, summarized)`, plus `.saved` (0.0 to 1.0) |
| `reset()` | clear the cached summary (new conversation) |

### Algorithm

1. If the chat already fits `max_tokens`, it is returned unchanged.
2. Leading `system` messages and the last `keep_last` messages are protected.
3. Older messages are dropped oldest-first until the rest fits the budget (minus `summary_tokens` if summarizing).
4. With `align_to_user`, the cut moves forward to the next `user` message, so an assistant tool call is never separated from its tool result.
5. With `summarize`, dropped messages are replaced by one message: `"Summary of the earlier conversation:\n..."`.

Protected messages are never dropped, so if they alone exceed `max_tokens`, the result is over budget. Check `last_stats.tokens_after`.

### Incremental summaries

A `Compressor` remembers the last summary. When the chat grows and more messages fall out of the window, the summarizer receives only:

```
["previous summary: <old summary>", "user: <newly dropped>", "assistant: <newly dropped>", ...]
```

That keeps summarization cost flat instead of growing with the chat. Use **one Compressor per conversation**.

### Message format

Each message is a dict with `role` and `content`. `content` may be a string or a list of content blocks (OpenAI / Anthropic style); only `{"type": "text"}` blocks are counted. Other keys (`tool_calls`, `name`, ...) are passed through untouched.

## Summarizers

Any function `list[str] -> str` works.

```python
from delbina import ollama_summarizer
summarize = ollama_summarizer("qwen2.5:1.5b", host="http://localhost:11434")
```

`ollama_summarizer` uses only the Python standard library and calls Ollama's `/api/generate`.

Custom example:

```python
def my_summarizer(lines: list[str]) -> str:
    return my_llm("Summarize briefly, keep facts and decisions:\n" + "\n".join(lines))
```

## Token counting

Default counting uses tiktoken's `cl100k_base`. That is exact for some OpenAI models and an approximation for others. For exact budgets pass your model's tokenizer:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
comp = Compressor(max_tokens=3000, count_tokens=lambda s: len(tok.encode(s)))
```

## Measurement tools

For research and benchmarks, delbina also works on plain lists of turns:

| Function | Meaning |
|---|---|
| `count(text)` | tokens in a string |
| `cumulative_cost(turns)` | total tokens if the full history is re-sent each turn |
| `cost(turns, strategy)` | total tokens under a strategy |
| `full` | send everything (baseline) |
| `last_n(n)` | last n turns |
| `head_tail(h, t)` | first h + last t turns |
| `rolling_summary(keep, summarize)` | summary of older turns + last `keep` |

## CLI

```
delbina compress FILE [--max-tokens N] [--keep-last N] [--ollama MODEL] [-o OUT]
delbina bench FILE [--ollama MODEL]
delbina --version
```

Equivalent: `python -m delbina ...`.
