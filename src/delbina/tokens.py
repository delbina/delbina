# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Token counting.

``count`` uses tiktoken's cl100k_base. ``counter(model)`` returns a counter for
a specific model family:

    counter("gpt-4o")                      # exact, tiktoken o200k_base
    counter("gpt-4")                       # exact, tiktoken cl100k_base
    counter("hf:Qwen/Qwen2.5-7B-Instruct") # exact, Hugging Face tokenizer (pip install tokenizers)
    counter("claude-sonnet-4")             # estimate (tokenizer not public)
    counter("llama3"), counter("qwen")     # estimate unless you use hf:<repo>
    counter("chars")                       # len(text) / 4, no dependencies, no network
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import tiktoken

TokenCounter = Callable[[str], int]


@lru_cache(maxsize=None)
def _encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    # Loaded lazily: tiktoken downloads the file on first use, so importing
    # delbina must not need network.
    return tiktoken.get_encoding(name)


def count(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def cumulative_cost(turns: list[str]) -> int:
    """Total tokens if the full history is re-sent each turn."""
    total, history = 0, 0
    for t in turns:
        history += count(t)
        total += history
    return total


def chars(text: str) -> int:
    """Rough estimate: 4 characters per token. No tokenizer, no network."""
    return (len(text) + 3) // 4


# Model families without a public tiktoken encoding. Ratios scale cl100k_base
# counts; they are rough averages for English text and code, not exact.
_ESTIMATES: dict[str, float] = {
    "claude": 1.15,
    "llama": 1.0,     # Llama 3 uses a 128k tiktoken-style vocab close to cl100k
    "qwen": 1.0,
    "mistral": 1.1,
    "mixtral": 1.1,
    "gemma": 1.0,
    "phi": 1.0,
    "deepseek": 1.0,
}

_O200K_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "o1", "o3", "o4", "chatgpt-4o", "gpt-oss")


def _tiktoken_counter(encoding: str) -> TokenCounter:
    def c(text: str) -> int:
        return len(_encoding(encoding).encode(text, disallowed_special=()))
    c.__name__ = f"tiktoken_{encoding}"
    return c


def _hf_counter(repo: str) -> TokenCounter:
    try:
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("hf:<repo> counters need `pip install tokenizers` (or delbina[hf])") from e
    tok = Tokenizer.from_pretrained(repo)

    def c(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)
    c.__name__ = f"hf_{repo}"
    return c


def counter(model: str) -> TokenCounter:
    """Return a token counter for a model name. See the module docstring."""
    name = model.strip()
    low = name.lower()
    if low == "chars":
        return chars
    if low.startswith("hf:"):
        return _hf_counter(name[3:])
    if low.startswith("tiktoken:"):
        return _tiktoken_counter(name[9:])
    low = low.split("/")[-1]  # "openai/gpt-4o" -> "gpt-4o"
    if low.startswith(_O200K_PREFIXES):
        return _tiktoken_counter("o200k_base")
    if low.startswith(("gpt-4", "gpt-3.5", "text-embedding")):
        return _tiktoken_counter("cl100k_base")
    for family, ratio in _ESTIMATES.items():
        if family in low:
            def c(text: str, _r: float = ratio) -> int:
                return round(count(text) * _r)
            c.__name__ = f"estimate_{family}"
            return c
    return count


def resolve(count_tokens: TokenCounter | str | None) -> TokenCounter:
    """Accept a counter function, a model name, or None (default cl100k_base)."""
    if count_tokens is None:
        return count
    if isinstance(count_tokens, str):
        return counter(count_tokens)
    return count_tokens
