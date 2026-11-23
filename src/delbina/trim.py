"""Tool-output helpers: find tool results in a message list and shorten them.

Supported shapes:
- OpenAI / Ollama / most agents: {"role": "tool", "content": "...", "tool_call_id": ...}
- legacy OpenAI functions:      {"role": "function", "name": ..., "content": "..."}
- Anthropic:                     {"role": "user", "content": [{"type": "tool_result", "content": ...}, ...]}
"""
from __future__ import annotations

import copy
from typing import Any, Callable

Message = dict[str, Any]
TokenCounter = Callable[[str], int]

TRIM_MARK = "\n[... delbina trimmed {n} tokens of tool output ...]\n"


def is_tool_result(message: Message) -> bool:
    if message.get("role") in ("tool", "function"):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def shorten(text: str, max_tokens: int, count_tokens: TokenCounter, head: float = 0.6) -> tuple[str, int]:
    """Keep the start and end of ``text`` within about ``max_tokens``.

    Returns (new_text, tokens_removed). The start of a tool output usually holds
    the structure (file header, first results), the end holds the outcome
    (errors, totals), so both are kept.
    """
    total = count_tokens(text)
    if total <= max_tokens or max_tokens <= 0:
        return text, 0
    chars_per_token = max(len(text) / max(total, 1), 1.0)
    budget = int(max_tokens * chars_per_token)
    h = int(budget * head)
    t = budget - h
    # Prefer cutting on line breaks so code and logs stay readable.
    h_end = text.rfind("\n", 0, h)
    h_end = h_end if h_end > h // 2 else h
    t_start = text.find("\n", len(text) - t)
    t_start = t_start + 1 if 0 <= t_start < len(text) - t // 2 else len(text) - t
    kept = text[:h_end] + text[t_start:]
    removed = max(total - count_tokens(kept), 0)
    return text[:h_end] + TRIM_MARK.format(n=removed) + text[t_start:], removed


def _shorten_value(value: Any, max_tokens: int, count_tokens: TokenCounter, text_blocks: bool = True) -> tuple[Any, int]:
    if isinstance(value, str):
        return shorten(value, max_tokens, count_tokens)
    if isinstance(value, list):
        removed = 0
        out = []
        for block in value:
            if text_blocks and isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text, r = shorten(block["text"], max_tokens, count_tokens)
                block = {**block, "text": text}
                removed += r
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                inner, r = _shorten_value(block.get("content", ""), max_tokens, count_tokens)
                block = {**block, "content": inner}
                removed += r
            out.append(block)
        return out, removed
    return value, 0


def trim_tool_message(message: Message, max_tokens: int, count_tokens: TokenCounter) -> tuple[Message, int]:
    """Return (possibly shortened copy, tokens_removed). Ids and other keys are kept."""
    if not is_tool_result(message):
        return message, 0
    # In Anthropic format the user's own text blocks sit next to tool_result
    # blocks; only the tool results are shortened there.
    text_blocks = message.get("role") in ("tool", "function")
    content, removed = _shorten_value(message.get("content", ""), max_tokens, count_tokens, text_blocks)
    if not removed:
        return message, 0
    out = copy.copy(message)
    out["content"] = content
    return out, removed
