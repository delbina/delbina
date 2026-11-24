# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Compression strategies.

A strategy maps the chat history so far (list of turns) to the context
actually sent to the model on that turn.
"""
from typing import Callable

from .tokens import count

Strategy = Callable[[list[str]], list[str]]
Summarizer = Callable[[list[str]], str]


def full(history: list[str]) -> list[str]:
    """Baseline: send everything."""
    return history


def last_n(n: int) -> Strategy:
    """Keep only the last n turns."""
    def s(history: list[str]) -> list[str]:
        return history[-n:]
    s.__name__ = f"last_{n}"
    return s


def head_tail(head: int, tail: int) -> Strategy:
    """Keep the first `head` turns (task setup) and the last `tail` turns."""
    def s(history: list[str]) -> list[str]:
        if len(history) <= head + tail:
            return history
        return history[:head] + history[-tail:]
    s.__name__ = f"head{head}_tail{tail}"
    return s


def naive_summarizer(turns: list[str], max_words: int = 12) -> str:
    """Placeholder: first few words of each turn. Swap for a local LLM later."""
    return " | ".join(" ".join(t.split()[:max_words]) for t in turns)


def rolling_summary(keep: int, summarize: Summarizer = naive_summarizer) -> Strategy:
    """Summarize everything except the last `keep` turns."""
    def s(history: list[str]) -> list[str]:
        if len(history) <= keep:
            return history
        old, recent = history[:-keep], history[-keep:]
        return ["[summary] " + summarize(old)] + recent
    s.__name__ = f"summary_keep{keep}"
    return s


def cost(turns: list[str], strategy: Strategy) -> int:
    """Total tokens sent across the chat under a strategy."""
    return sum(
        sum(count(t) for t in strategy(turns[: i + 1]))
        for i in range(len(turns))
    )
