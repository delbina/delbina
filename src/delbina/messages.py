"""Compress chat message lists before they are sent to a model.

Works with the common chat format used by OpenAI, Ollama, Anthropic and most
local agents: a list of dicts with "role" and "content".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

Message = dict[str, Any]
TokenCounter = Callable[[str], int]
Summarizer = Callable[[list[str]], str]

SUMMARY_PREFIX = "Summary of the earlier conversation:\n"
MESSAGE_OVERHEAD = 4  # rough per-message cost of role/formatting tokens


def message_text(message: Message) -> str:
    """Text of a message. Handles plain strings and lists of content blocks."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "" if content is None else str(content)


@dataclass(frozen=True)
class Stats:
    """What the last compress() call did."""

    tokens_before: int
    tokens_after: int
    dropped: int
    summarized: bool

    @property
    def saved(self) -> float:
        """Fraction of tokens saved, 0.0 to 1.0."""
        if self.tokens_before == 0:
            return 0.0
        return 1 - self.tokens_after / self.tokens_before


class Compressor:
    """Keeps a chat under a token budget.

    Leading system messages and the last ``keep_last`` messages are always
    kept. Older messages are dropped until the chat fits ``max_tokens``. If a
    ``summarize`` function is given, dropped messages are replaced by one
    summary message.

    The summary is cached: on the next call, only newly dropped messages are
    sent to the summarizer together with the previous summary, so a long chat
    is not re-summarized from scratch every turn. Use one Compressor per
    conversation.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        keep_last: int = 6,
        summarize: Summarizer | None = None,
        count_tokens: TokenCounter | None = None,
        summary_role: str = "system",
        summary_tokens: int = 300,
        align_to_user: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        if count_tokens is None:
            from .tokens import count as count_tokens
        self.max_tokens = max_tokens
        self.keep_last = keep_last
        self.summarize = summarize
        self.count_tokens = count_tokens
        self.summary_role = summary_role
        self.summary_tokens = summary_tokens if summarize else 0
        self.align_to_user = align_to_user
        self.last_stats: Stats | None = None
        self._cache_key: tuple[tuple[str, str], ...] = ()
        self._cache_summary: str = ""

    def tokens(self, messages: Iterable[Message]) -> int:
        """Approximate token count of a message list."""
        return sum(self.count_tokens(message_text(m)) + MESSAGE_OVERHEAD for m in messages)

    def compress(self, messages: list[Message]) -> list[Message]:
        """Return a new message list that fits the budget. Input is not modified."""
        msgs = list(messages)
        sizes = [self.count_tokens(message_text(m)) + MESSAGE_OVERHEAD for m in msgs]
        before = sum(sizes)

        n_system = 0
        while n_system < len(msgs) and msgs[n_system].get("role") == "system":
            n_system += 1
        system, rest, rest_sizes = msgs[:n_system], msgs[n_system:], sizes[n_system:]
        protected_from = max(len(rest) - self.keep_last, 0)

        if before <= self.max_tokens or protected_from == 0:
            self.last_stats = Stats(before, before, 0, False)
            return msgs

        budget = self.max_tokens - sum(sizes[:n_system]) - self.summary_tokens
        kept = sum(rest_sizes)
        cut = 0
        while cut < protected_from and kept > budget:
            kept -= rest_sizes[cut]
            cut += 1

        # Start the kept part on a user message so tool calls/results and
        # user/assistant alternation are not split mid-exchange.
        if self.align_to_user and cut > 0:
            while cut < protected_from and rest[cut].get("role") != "user":
                cut += 1

        if cut == 0:
            self.last_stats = Stats(before, before, 0, False)
            return msgs

        dropped, kept_msgs = rest[:cut], rest[cut:]
        out = list(system)
        summarized = False
        if self.summarize is not None:
            out.append({"role": self.summary_role, "content": SUMMARY_PREFIX + self._summary_for(dropped)})
            summarized = True
        out.extend(kept_msgs)

        self.last_stats = Stats(before, self.tokens(out), len(dropped), summarized)
        return out

    def _summary_for(self, dropped: list[Message]) -> str:
        key = tuple((str(m.get("role", "")), message_text(m)) for m in dropped)
        cached = self._cache_key
        if key == cached:
            return self._cache_summary
        if cached and key[: len(cached)] == cached:
            new = [f"{r}: {t}" for r, t in key[len(cached):]]
            summary = self.summarize([f"previous summary: {self._cache_summary}", *new])
        else:
            summary = self.summarize([f"{r}: {t}" for r, t in key])
        self._cache_key, self._cache_summary = key, summary
        return summary

    def reset(self) -> None:
        """Forget the cached summary (start of a new conversation)."""
        self._cache_key, self._cache_summary = (), ""
        self.last_stats = None


def compress(messages: list[Message], **options: Any) -> list[Message]:
    """One-shot compression. For multi-turn chats prefer a Compressor instance,
    which caches the summary between turns. Options are Compressor's arguments."""
    return Compressor(**options).compress(messages)
