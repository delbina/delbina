# Integrations

Two ways to use delbina:

- **Proxy**: run `delbina serve` and point the client at it. No code changes. Best for agents and tools you didn't write.
- **Library**: keep the full history in your own list and send `comp.compress(history)` to the model. Best when you own the loop.

## Proxy

```bash
delbina serve --upstream https://api.openai.com/v1 --max-tokens 16000
```

### Anything built on the OpenAI SDK

```bash
# macOS / Linux
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
# Windows PowerShell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8787/v1"
```

Or in code:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1")
```

### Anthropic SDK

```bash
delbina serve --upstream https://api.anthropic.com/v1 --max-tokens 30000
```

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8787")
```

The summary (if enabled) is appended to `system`; existing `cache_control` blocks are kept. For summaries with Anthropic as the upstream, use `--ollama MODEL`; `--summary-model` expects an OpenAI-compatible upstream.

### Local models (Ollama, LM Studio, vLLM, llama.cpp)

```bash
delbina serve --upstream http://localhost:11434/v1 --max-tokens 6000 --ollama qwen2.5:1.5b --background
```

Point the client at `http://127.0.0.1:8787/v1` (OpenAI-compatible) or `http://127.0.0.1:8787/api/chat` (Ollama native). Keeping the prompt short matters even more locally: prompt processing time grows with context length, and llama.cpp/Ollama also reuse the cached prefix between calls.

### Many conversations at once

Send `X-Delbina-Session: <conversation id>` with each request so every conversation gets its own compressor even if they share a system prompt and first message:

```python
client.chat.completions.create(model=..., messages=..., extra_headers={"X-Delbina-Session": chat_id})
```

## Library

### OpenAI (and OpenAI-compatible servers)

```python
from openai import OpenAI
from delbina import Compressor

client = OpenAI()
comp = Compressor(max_tokens=16000, count_tokens="gpt-4o")
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=comp.compress(history))
    text = resp.choices[0].message.content
    history.append({"role": "assistant", "content": text})
    return text
```

### Anthropic

Anthropic takes the system prompt as a separate parameter, so split it out after compressing. The summary (role `system`) ends up in the system prompt too.

```python
import anthropic
from delbina import Compressor

client = anthropic.Anthropic()
comp = Compressor(max_tokens=30000, count_tokens="claude")
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    out = comp.compress(history)
    system = "\n\n".join(m["content"] for m in out if m["role"] == "system")
    msgs = [m for m in out if m["role"] != "system"]
    resp = client.messages.create(model="YOUR_MODEL", max_tokens=1024, system=system, messages=msgs)
    text = resp.content[0].text
    history.append({"role": "assistant", "content": text})
    return text
```

Anthropic only caches what you mark: put `cache_control` on the system prompt and/or the last message. delbina keeps the prefix stable between compaction events, so those cache entries keep hitting.

### Tool-calling agent

See [examples/agent_loop.py](../examples/agent_loop.py). The essentials:

```python
comp = Compressor(max_tokens=24000, keep_last=8, tool_max_tokens=400, keep_tool_results=2)
comp.load(state_file)                       # survive restarts

while not done:
    resp = llm(comp.compress(history))      # history keeps every tool result in full
    history.append(resp.message)
    for call in resp.tool_calls:
        history.append({"role": "tool", "tool_call_id": call.id, "content": run_tool(call)})
    comp.save(state_file)
```

### Async apps

```python
comp = Compressor(max_tokens=16000, summarize=my_async_summarizer)
messages_to_send = await comp.acompress(history)
```

### Ollama (local, free)

See [examples/ollama_chat.py](../examples/ollama_chat.py): a complete terminal chat with no dependencies besides delbina.

```bash
ollama pull qwen2.5:1.5b
python examples/ollama_chat.py --model qwen2.5:1.5b --max-tokens 1500
```
