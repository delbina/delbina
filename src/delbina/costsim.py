"""Replay a saved conversation and estimate what the input tokens would cost,
with provider prompt caching taken into account.

A model call is assumed after every user or tool message. For each call the
prompt is compared with the previous call's prompt: the identical leading
messages count as cached (if at least ``min_prefix`` tokens, like OpenAI and
Anthropic require), the rest as uncached.

    billed = uncached + cached * cache_price

``cache_price`` is the cached-token price relative to normal input:
0.1 for Anthropic cache reads, OpenAI GPT-4.1/GPT-5 family and DeepSeek;
0.5 for older OpenAI models (GPT-4o). Cache *write* surcharges (Anthropic
1.25x) are ignored, which slightly favours the uncompressed baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .messages import MESSAGE_OVERHEAD, Compressor, fingerprint, message_text
from .tokens import TokenCounter, resolve


@dataclass(frozen=True)
class Replay:
    name: str
    calls: int
    input_tokens: int
    cached_tokens: int
    compactions: int
    cache_price: float
    avg_turns_kept: float = 0.0  # user turns visible to the model per call (how much context survives)

    @property
    def uncached_tokens(self) -> int:
        return self.input_tokens - self.cached_tokens

    @property
    def billed(self) -> float:
        return self.uncached_tokens + self.cached_tokens * self.cache_price


def call_points(messages: list[dict]) -> list[int]:
    """Indexes after which the agent would call the model."""
    return [i + 1 for i, m in enumerate(messages) if m.get("role") in ("user", "tool", "function")]


def replay(
    messages: list[dict],
    name: str,
    make: Callable[[], Compressor] | None,
    count_tokens: TokenCounter | str | None = None,
    cache_price: float = 0.1,
    min_prefix: int = 1024,
) -> Replay:
    counter = resolve(count_tokens)
    sizes: dict[str, int] = {}

    def size(m: dict) -> tuple[str, int]:
        fp = fingerprint(m)
        if fp not in sizes:
            sizes[fp] = counter(message_text(m)) + MESSAGE_OVERHEAD
        return fp, sizes[fp]

    comp = make() if make else None
    prev: list[str] = []
    total = cached = compactions = calls = turns_kept = 0
    for end in call_points(messages):
        prompt = comp.compress(messages[:end]) if comp else messages[:end]
        if comp is not None and comp.last_stats.compacted:
            compactions += 1
        fs = [size(m) for m in prompt]
        prefix = 0
        for (fp, n), old in zip(fs, prev):
            if fp != old:
                break
            prefix += n
        tokens = sum(n for _, n in fs)
        total += tokens
        cached += prefix if prefix >= min_prefix else 0
        prev = [fp for fp, _ in fs]
        calls += 1
        turns_kept += sum(1 for m in prompt if m.get("role") == "user")
    return Replay(name, calls, total, cached, compactions, cache_price, turns_kept / max(calls, 1))


def compare(
    messages: list[dict],
    max_tokens: int,
    keep_last: int = 6,
    count_tokens: TokenCounter | str | None = None,
    cache_price: float = 0.1,
    min_prefix: int = 1024,
    summarize=None,
    target_ratio: float = 0.6,
    tool_max_tokens: int | None = 400,
    keep_tool_results: int = 2,
) -> list[Replay]:
    """Full history vs sliding window (delbina 0.2 behaviour) vs delbina."""
    counter = resolve(count_tokens)
    common = dict(max_tokens=max_tokens, keep_last=keep_last, count_tokens=counter, summarize=summarize)
    ours = dict(target_ratio=target_ratio, keep_tool_results=keep_tool_results)
    runs: list[tuple[str, Callable[[], Compressor] | None]] = [
        ("full history", None),
        ("sliding window", lambda: Compressor(**common, target_ratio=1.0, tool_max_tokens=None)),
        ("delbina, no tool trim", lambda: Compressor(**common, **ours, tool_max_tokens=None)),
        ("delbina", lambda: Compressor(**common, **ours, tool_max_tokens=tool_max_tokens)),
    ]
    return [replay(messages, n, m, counter, cache_price, min_prefix) for n, m in runs]
