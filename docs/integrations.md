# Integrations

delbina only transforms a list of messages, so it works with any client. Keep the **full** history in your own list and send `comp.compress(history)` to the model.

## OpenAI (and OpenAI-compatible servers: vLLM, LM Studio, llama.cpp server, Groq, ...)

```python
from openai import OpenAI
from delbina import Compressor

client = OpenAI()  # reads OPENAI_API_KEY
comp = Compressor(max_tokens=3000, keep_last=6)
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(model="YOUR_MODEL", messages=comp.compress(history))
    text = resp.choices[0].message.content
    history.append({"role": "assistant", "content": text})
    return text
```

## Anthropic

Anthropic takes the system prompt as a separate parameter, so split it out after compressing. The summary (role `system`) ends up in the system prompt too.

```python
import anthropic
from delbina import Compressor

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
comp = Compressor(max_tokens=3000, keep_last=6)
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

## Ollama (local, free)

See [examples/ollama_chat.py](../examples/ollama_chat.py): a complete terminal chat with no dependencies besides delbina.

```bash
ollama pull qwen2.5:1.5b
python examples/ollama_chat.py --model qwen2.5:1.5b --max-tokens 1500
```

## Your own agent

```python
from delbina import Compressor

class Agent:
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.history = [{"role": "system", "content": system_prompt}]
        self.comp = Compressor(max_tokens=4000, keep_last=8)

    def step(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        reply = self.llm(self.comp.compress(self.history))
        self.history.append({"role": "assistant", "content": reply})
        return reply
```

Tool-calling agents: keep `align_to_user=True` (the default) so an assistant `tool_calls` message is never kept without its matching `tool` result.
