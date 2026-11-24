# Getting started

This guide takes you from nothing to a working setup, step by step. Pick your path:

- **Path A: free tier** (Gemini, Groq, OpenRouter free models, ...). Your limit is *tokens per minute* and *requests per day*. delbina makes each request smaller, so you hit rate limits less often and your daily quota lasts longer.
- **Path B: paid API** (OpenAI, Anthropic, DeepSeek, ...) for agents and long chats. Your limit is the *bill*. delbina cuts input tokens and keeps prompt caching working, so the tokens you do send are mostly billed at the cached discount.

Both paths use the same two building blocks:

| | Proxy (`delbina serve`) | Library (`Compressor`) |
|---|---|---|
| You change | one URL (or one environment variable) | a few lines in your code |
| Works with | any app, agent or tool that lets you set an OpenAI-compatible base URL | your own Python code |
| Use when | you didn't write the agent, or you don't want to touch its code | you own the chat/agent loop |

---

## Step 1: install

Python 3.10 or newer.

```bash
pip install delbina
python -m delbina --version
```

The only dependency is `tiktoken` (a few MB). On first use it downloads a ~2 MB tokenizer file, so the first run needs internet. Nothing else is downloaded: delbina does **not** need a local model or a GPU.

> Windows: if `delbina.exe` is blocked by Smart App Control, always use `python -m delbina ...` instead of `delbina ...`. Every command in this guide works both ways.

---

## Step 2: get your API details

You need three things from your provider: the **base URL**, an **API key**, and a **model ID**.

| Provider | Base URL (`--upstream`) | Key from | Notes |
|---|---|---|---|
| Google Gemini (free tier available) | `https://generativelanguage.googleapis.com/v1beta/openai` | Google AI Studio → Get API key | copy the model ID from AI Studio; free-tier prompts may be used by Google to improve its products |
| Groq (free tier available) | `https://api.groq.com/openai/v1` | console.groq.com | |
| OpenRouter (some free models) | `https://openrouter.ai/api/v1` | openrouter.ai/keys | free models end in `:free` |
| OpenAI | `https://api.openai.com/v1` | platform.openai.com | |
| Anthropic | `https://api.anthropic.com/v1` | console.anthropic.com | use the Anthropic SDK (`/v1/messages`) |
| DeepSeek | `https://api.deepseek.com/v1` | platform.deepseek.com | |
| Local (Ollama, LM Studio, vLLM) | e.g. `http://localhost:11434/v1` | none | |

Model IDs and free-tier limits change often. Always copy the current model ID from your provider's console.

Put the key in an environment variable instead of in commands or code, so it does not end up in your shell history or a git commit:

```powershell
# Windows PowerShell (this window only)
$env:OPENAI_API_KEY = "your-key"
```

```bash
# macOS / Linux
export OPENAI_API_KEY="your-key"
```

(The variable is called `OPENAI_API_KEY` because OpenAI-compatible tools read that name, even when the key is for Gemini, Groq or OpenRouter.)

---

## Step 3: start the proxy

Open a terminal and leave it running:

```bash
python -m delbina serve --upstream <BASE_URL> --max-tokens <BUDGET>
```

Examples:

```bash
# Path A: Gemini free tier
python -m delbina serve --upstream https://generativelanguage.googleapis.com/v1beta/openai --max-tokens 16000

# Path B: OpenAI, an agent with large tool outputs
python -m delbina serve --upstream https://api.openai.com/v1 --max-tokens 32000
```

You should see:

```
delbina 0.4.0 proxy on http://127.0.0.1:8787/v1 -> https://...
budget 16000 tokens, target 60%, stats at http://127.0.0.1:8787/delbina/stats
```

The proxy listens only on your own machine (`127.0.0.1`). Your API key is passed through to the provider in the request header. delbina does not store it or log it.

### Choosing `--max-tokens`

This is the most important setting. It is the most context the model will ever see.

| Situation | Start with |
|---|---|
| Free tier, simple chat | 6000 to 12000 |
| Free tier, agent with tools | 12000 to 24000 |
| Paid, chat assistant | 8000 to 16000 |
| Paid, coding agent (reads files, runs tests) | 24000 to 64000 |

Rules of thumb:

- It must be clearly larger than your biggest single tool output and your system prompt combined. Otherwise delbina has to compact on almost every request, which defeats prompt caching.
- Lower = cheaper and fewer rate-limit errors, but the model forgets older parts of the conversation sooner. Step 7 shows how to measure that instead of guessing.

---

## Step 4: point your app or agent at the proxy

Replace the provider's base URL with `http://127.0.0.1:8787/v1`. Keep your API key and model name exactly as before.

**Anything built on the official OpenAI SDK** (Python or JavaScript) reads an environment variable, so often this is all you need:

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8787/v1"
python your_agent.py
```

**OpenAI SDK in code:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1")   # key still read from OPENAI_API_KEY
```

**LangChain:**

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="MODEL_ID", base_url="http://127.0.0.1:8787/v1")
```

**Anthropic SDK** (start the proxy with `--upstream https://api.anthropic.com/v1`):

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8787")
```

**Apps and tools with a settings screen** (coding assistants, chat UIs, automation tools): look for a setting named *OpenAI-compatible*, *custom provider*, *base URL* or *API endpoint*, and enter `http://127.0.0.1:8787/v1`.

**Several conversations at once** (for example an agent framework running many tasks): send a header with a unique id per conversation, so each gets its own compressor:

```python
client.chat.completions.create(model=..., messages=..., extra_headers={"X-Delbina-Session": task_id})
```

---

## Step 5: check that it works

1. Use your app normally for a while.
2. Watch the proxy terminal. Each request prints a line like:

   ```
   [delbina] gpt-4o-mini: 41250 -> 18903 tokens (54% saved), compacted, 6 tool outputs trimmed
   [delbina] gpt-4o-mini: 42100 -> 19750 tokens (53% saved)
   ```

   - Lines **without** `compacted` are the cheap ones: the beginning of the prompt is unchanged, so the provider can serve it from cache.
   - `compacted` should appear **occasionally**, not on every line (see Troubleshooting if it does).
3. Open `http://127.0.0.1:8787/delbina/stats` in a browser for totals:

   ```json
   {"requests": 134, "compressed_requests": 134, "compactions": 5, "tokens_before": 7160495, "tokens_after": 2901322, "saved": 0.5948}
   ```

4. Compare with your provider's usage page (OpenAI, Anthropic, Google AI Studio). On paid APIs, look at the **cached input tokens** figure as well. It should be a large share of input.

---

## Step 6: optional, keep a summary of old messages

By default, messages that fall out of the budget are simply dropped (old tool outputs are shortened first). If your agent needs to remember facts from early in a long session, have delbina write a running summary.

**Path A (free tier):** use the same provider and a small, fast model. The summary is made with your own key:

```bash
python -m delbina serve --upstream https://generativelanguage.googleapis.com/v1beta/openai `
  --max-tokens 16000 --summary-model MODEL_ID --background
```

**Path B (paid):** use the cheapest model of your provider:

```bash
python -m delbina serve --upstream https://api.openai.com/v1 --max-tokens 32000 --summary-model gpt-4o-mini --background
```

- A summary is made **once per compaction event**, not on every request, so it adds few calls. On a free tier those calls still count towards your requests per day.
- `--background` prepares the summary before it is needed, so your replies are not delayed.
- `--summary-model` needs an OpenAI-compatible upstream. With Anthropic as the upstream, use `--ollama MODEL` (a local model) or no summary.

---

## Step 7: measure before you trust it

Savings only count if answers stay correct. delbina has two built-in measurements.

**Cost** (no API calls, free). Replays a saved conversation and estimates the bill with prompt caching:

```bash
python -m delbina cost my_chat.json --max-tokens 32000 --cache-price 0.1
```

`my_chat.json` is `{"messages": [...]}` in the usual chat format. Use `--cache-price 0.1` for Anthropic and recent OpenAI models, `0.5` for older OpenAI models. Try a few `--max-tokens` values and pick the cheapest one that still keeps enough turns visible (the "turns kept" column).

**Quality** (uses API calls). Plants facts in long agent conversations, then asks the model for them:

```bash
# free tier: stay under the rate limit, keep the run small
python -m delbina eval --base-url https://generativelanguage.googleapis.com/v1beta/openai `
  --model MODEL_ID --summary-model MODEL_ID --reasoning-effort minimal --rpm 8 --cases 12

# paid
python -m delbina eval --base-url https://api.openai.com/v1 --model gpt-4o-mini --summary-model gpt-4o-mini
```

The table shows, for each mode, how often the model still answered correctly, split by whether the fact was early, in the middle or late in the conversation. If `delbina` loses early facts and `summary` keeps them, turn on summaries (step 6). If the daily quota runs out, split the run: `--modes full,sliding` one day, `--modes delbina,summary` the next.

---

## Step 8: library instead of proxy (your own code)

If you write the agent loop yourself, skip the proxy:

```python
from delbina import Compressor

comp = Compressor(
    max_tokens=24000,
    count_tokens="gpt-4o",        # or "claude", "chars", "hf:<repo>"
    # summarize=openai_summarizer("gpt-4o-mini"),   # optional, step 6
)

history = [{"role": "system", "content": "You are a coding agent."}]
# ... in your loop:
response = client.chat.completions.create(model=MODEL, messages=comp.compress(history))
```

- Keep the **full** history in your own list. Send only `comp.compress(history)`.
- Use **one `Compressor` per conversation**, and keep it alive between turns. That is what keeps the prefix stable for caching.
- To survive restarts, save the history and `comp.state()` together, and on startup restore both (see [examples/agent_loop.py](../examples/agent_loop.py)).
- In async code use `await comp.acompress(history)`.

Full reference: [guide.md](guide.md). More integrations: [integrations.md](integrations.md).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `compacted` on almost every request | The budget is too small for what must always be kept (system prompt, current request, newest tool outputs). Raise `--max-tokens`, or lower `--tool-max-tokens` / `--keep-tool-results`. |
| `compacted` every request and savings near 0 | Your app changes the system prompt or early messages on every call (for example, it puts the current time in the system prompt). delbina then starts fresh each time. Move changing content to the latest user message if you can. |
| 429 errors (rate limit) | Free tier limit hit. Lower `--max-tokens` (fewer tokens per minute) and slow your app down. The provider's limits are per project, not per key. |
| `delbina: upstream unreachable` (502) | Wrong `--upstream`, no internet, or the provider is down. Test the URL without the proxy first. |
| 401 / 403 from the provider | The key is missing or wrong. The proxy passes your key through unchanged, so check the app's key setting. |
| 404 from the provider | Wrong model ID or wrong base URL path. Copy both from the provider's console. |
| The model forgot something from early in the chat | Expected without a summary: turn on step 6, or raise `--max-tokens`. Measure with step 7. |
| Empty answers from Gemini or other reasoning models | Thinking tokens count against the output cap. Give the app a larger `max_tokens`, or use a non-thinking model. |
| Port 8787 already in use | `--port 8788`, and use that port in the base URL. |
| Want it reachable from other devices | `--host 0.0.0.0`, but then anyone on your network can use it, and with key-less upstreams (Ollama) that means your model. Prefer keeping `127.0.0.1`. |
