# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Compress chat message lists before they are sent to a model.

Works with the common chat format used by OpenAI, Ollama, Anthropic and most
local agents: a list of dicts with "role" and "content".

Prompt caching
--------------
Providers (OpenAI, Anthropic, DeepSeek, Gemini, vLLM, llama.cpp ...) charge far
less for, and answer faster on, a prompt whose beginning is byte-identical to
an earlier one. A compressor that drops one old message every turn changes the
beginning every turn and loses that discount.

``Compressor`` therefore works in *compaction events*. While the chat fits
``max_tokens`` the output only grows at the end. When it overflows, delbina
trims old tool outputs and drops/summarizes old messages down to
``target_ratio * max_tokens`` in one step, then lets the chat grow again. The
prefix stays identical between events, so it stays cached.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .tokens import TokenCounter, resolve
from .trim import is_tool_result, trim_tool_message

Message = dict[str, Any]
Summarizer = Callable[[list[str]], str]

SUMMARY_PREFIX = "Summary of the earlier conversation:\n"
MESSAGE_OVERHEAD = 4  # rough per-message cost of role/formatting tokens
STATE_VERSION = 1


def message_text(message: Message) -> str:
    """Text of a message: plain strings, text blocks, tool results, tool calls."""
    parts: list[str] = []
    content = message.get("content", "")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(str(block.get("text", "")))
            elif kind == "tool_result":
                parts.append(message_text({"content": block.get("content", "")}))
            elif kind == "tool_use":
                parts.append(f"{block.get('name', '')}({json.dumps(block.get('input', {}), ensure_ascii=False)})")
    elif content is not None:
        parts.append(str(content))
    for call in message.get("tool_calls") or ():
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
    return "\n".join(p for p in parts if p)


def fingerprint(message: Message) -> str:
    raw = json.dumps(message, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()


def is_turn_start(message: Message) -> bool:
    """A user message that is not a tool result: a safe place to start the kept part."""
    return message.get("role") == "user" and not is_tool_result(message)


@dataclass(frozen=True)
class Stats:
    """What the last compress() call did."""

    tokens_before: int
    tokens_after: int
    dropped: int
    summarized: bool
    trimmed: int = 0          # tool outputs shortened in the result
    compacted: bool = False   # True if this call changed the prefix (cache miss expected)

    @property
    def saved(self) -> float:
        """Fraction of tokens saved, 0.0 to 1.0."""
        if self.tokens_before == 0:
            return 0.0
        return 1 - self.tokens_after / self.tokens_before


class Compressor:
    """Keeps one conversation under a token budget, cache-friendly.

    Args:
        max_tokens: budget for the returned list.
        keep_last: the last N messages are never dropped.
        summarize: ``list[str] -> str`` (sync) or ``async`` function. None = drop only.
        count_tokens: ``str -> int``, or a model name such as ``"gpt-4o"`` (see ``delbina.tokens.counter``).
        target_ratio: on overflow, compress to this fraction of ``max_tokens``.
            Lower = fewer compaction events (better caching), less context. 1.0 = old
            behaviour (cut just enough every turn, breaks prompt caching).
        tool_max_tokens: old tool outputs are shortened to about this many tokens.
            None disables trimming.
        keep_tool_results: the most recent N tool outputs are never shortened.
        background: prepare the next summary in a background thread before the
            budget is hit, so summarizing does not delay the reply.
        prefetch_ratio: with ``background``, start preparing at this fraction of ``max_tokens``.

    Leading system messages are always kept. The kept part always starts on a
    user message, so tool calls are never separated from their results.
    Use one Compressor per conversation.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        keep_last: int = 6,
        summarize: Callable[[list[str]], Any] | None = None,
        count_tokens: TokenCounter | str | None = None,
        summary_role: str = "system",
        summary_tokens: int = 300,
        align_to_user: bool = True,
        target_ratio: float = 0.6,
        tool_max_tokens: int | None = 400,
        keep_tool_results: int = 2,
        background: bool = False,
        prefetch_ratio: float = 0.85,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        if not 0 < target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if not 0 < prefetch_ratio <= 1:
            raise ValueError("prefetch_ratio must be in (0, 1]")
        if keep_tool_results < 0:
            raise ValueError("keep_tool_results must be >= 0")
        self.max_tokens = max_tokens
        self.keep_last = keep_last
        self.summarize = summarize
        self.count_tokens = resolve(count_tokens)
        self.summary_role = summary_role
        self.summary_tokens = summary_tokens if summarize else 0
        self.align_to_user = align_to_user
        self.target_ratio = target_ratio
        self.tool_max_tokens = tool_max_tokens
        self.keep_tool_results = keep_tool_results
        self.background = background
        self.prefetch_ratio = prefetch_ratio
        self.last_stats: Stats | None = None

        self._lock = threading.RLock()
        self._size_cache: dict[str, int] = {}
        self._trim_cache: dict[str, Message] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._pending: tuple[int, str, list[str], int, Future] | None = None
        self._n_trimmed = 0
        self._out_tokens = 0
        self._clear_state()

    # ---------------------------------------------------------------- public

    def tokens(self, messages: Iterable[Message]) -> int:
        """Approximate token count of a message list."""
        return sum(self.count_tokens(message_text(m)) + MESSAGE_OVERHEAD for m in messages)

    def compress(self, messages: list[Message]) -> list[Message]:
        """Return a list that fits the budget. Input is not modified."""
        if self.summarize is not None and inspect.iscoroutinefunction(self.summarize):
            raise TypeError("summarize is async: use `await compressor.acompress(messages)`")
        return self._compress(messages, self.summarize)

    async def acompress(self, messages: list[Message]) -> list[Message]:
        """Async version. Accepts sync or async summarizers; never blocks the event loop."""
        fn = self.summarize
        if fn is not None and inspect.iscoroutinefunction(fn):
            loop = asyncio.get_running_loop()
            async_fn = fn

            def fn(lines: list[str]) -> str:  # runs in a worker thread
                return asyncio.run_coroutine_threadsafe(async_fn(lines), loop).result()
        return await asyncio.to_thread(self._compress, messages, fn)

    def reset(self) -> None:
        """Forget all state (start of a new conversation)."""
        with self._lock:
            self._clear_state()
            self.last_stats = None

    @property
    def summary(self) -> str:
        """Current running summary ("" if none)."""
        return self._summary

    def state(self) -> dict[str, Any]:
        """JSON-serializable state, to persist across restarts. See load_state()."""
        with self._lock:
            return {
                "version": STATE_VERSION,
                "system": self._system_fp,
                "hashes": list(self._hashes),
                "cut": self._cut,
                "trim_until": self._trim_until,
                "summary": self._summary,
            }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state from state(). Ignored safely if the chat no longer matches."""
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported delbina state version: {state.get('version')!r}")
        with self._lock:
            self._clear_state()
            self._system_fp = state.get("system")
            self._hashes = list(state.get("hashes", []))
            self._cut = int(state.get("cut", 0))
            self._trim_until = int(state.get("trim_until", 0))
            self._summary = str(state.get("summary", ""))
            self._active = bool(self._hashes)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state()), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        """Load state saved with save(). Returns False if the file does not exist."""
        p = Path(path)
        if not p.exists():
            return False
        self.load_state(json.loads(p.read_text(encoding="utf-8")))
        return True

    def close(self) -> None:
        """Stop the background thread, if any."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # --------------------------------------------------------------- internals

    def _clear_state(self) -> None:
        self._active = False
        self._system_fp: str | None = None
        self._hashes: list[str] = []
        self._cut = 0
        self._trim_until = 0
        self._summary = ""
        self._pending = None

    def _size(self, message: Message, fp: str) -> int:
        size = self._size_cache.get(fp)
        if size is None:
            if len(self._size_cache) > 50_000:
                self._size_cache.clear()
            size = self._size_cache[fp] = self.count_tokens(message_text(message)) + MESSAGE_OVERHEAD
        return size

    def _trimmed(self, message: Message, fp: str) -> Message:
        if self.tool_max_tokens is None or not is_tool_result(message):
            return message
        out = self._trim_cache.get(fp)
        if out is None:
            if len(self._trim_cache) > 10_000:
                self._trim_cache.clear()
            out, _ = trim_tool_message(message, self.tool_max_tokens, self.count_tokens)
            self._trim_cache[fp] = out
        return out

    def _cap_summary(self, summary: Any) -> str:
        """Keep a summary within ``summary_tokens``: the planner reserved exactly that much
        room, so a longer summary would overflow the budget and force early compactions."""
        text = str(summary or "").strip()
        limit = self.summary_tokens
        if not limit or self.count_tokens(text) <= limit:
            return text
        total = self.count_tokens(text)
        text = text[: max(int(len(text) * limit / total), 1)]
        while text and self.count_tokens(text) > limit:
            text = text[: int(len(text) * 0.9)]
        space = text.rfind(" ")
        return (text[:space] if space > len(text) // 2 else text).rstrip()

    def _summary_message(self, summary: str) -> Message:
        return {"role": self.summary_role, "content": SUMMARY_PREFIX + summary}

    def _eff_size(self, rest, fps, sizes, i: int, trim_until: int) -> int:
        if i < trim_until and self.tool_max_tokens is not None and is_tool_result(rest[i]):
            return self._size(self._trimmed(rest[i], fps[i]), "t" + fps[i])
        return sizes[i]

    def _build(self, system, rest, fps, cut, trim_until, summary, system_size=0) -> list[Message]:
        """Assemble the output; also sets self._out_tokens and self._n_trimmed (cached sizes)."""
        out = list(system)
        total = system_size
        if summary:
            sm = self._summary_message(summary)
            out.append(sm)
            total += self.count_tokens(sm["content"]) + MESSAGE_OVERHEAD
        self._n_trimmed = 0
        for i in range(cut, len(rest)):
            m = self._trimmed(rest[i], fps[i]) if i < trim_until else rest[i]
            if m is rest[i]:
                total += self._size(m, fps[i])
            else:
                self._n_trimmed += 1
                total += self._size(m, "t" + fps[i])
            out.append(m)
        self._out_tokens = total
        return out

    def _plan(self, system_size, rest, fps, sizes, cut0, trim0) -> tuple[int, int]:
        """Where a compaction now would cut and trim: (cut, trim_until)."""
        n = len(rest)
        trim_until = trim0
        if self.tool_max_tokens is not None:
            tools = [i for i in range(n) if is_tool_result(rest[i])]

            def until(keep: int) -> int:
                if len(tools) <= keep:
                    return trim0
                return max(trim0, tools[-keep] if keep else n)

            trim_until = until(self.keep_tool_results)
            if self.keep_tool_results > 1:
                # If the newest tool outputs alone overflow the budget, keep only the very newest intact.
                tail = system_size + self.summary_tokens + sum(
                    self._eff_size(rest, fps, sizes, i, trim_until) for i in range(trim_until, n))
                if tail > self.max_tokens * self.target_ratio:
                    trim_until = until(1)
        eff = [self._eff_size(rest, fps, sizes, i, trim_until) for i in range(n)]
        target = int(self.max_tokens * self.target_ratio)
        reserve = self.summary_tokens
        kept = sum(eff[cut0:])
        if system_size + reserve + kept <= target:
            return cut0, trim_until  # trimming tool outputs was enough

        # The newest user turn (the current request and its tool calls) is never dropped.
        last_start = next((i for i in range(n - 1, -1, -1) if is_turn_start(rest[i])), n)
        protected_from = max(min(n - self.keep_last, last_start), cut0)
        budget = target - system_size - reserve
        cut = cut0
        while cut < protected_from and kept > budget:
            kept -= eff[cut]
            cut += 1
        if self.align_to_user and cut0 < cut < n and not is_turn_start(rest[cut]):
            fwd = cut
            while fwd < protected_from and not is_turn_start(rest[fwd]):
                fwd += 1
            if is_turn_start(rest[fwd]):
                cut = fwd
            else:  # no safe start ahead without dropping protected messages: cut earlier
                back = cut - 1
                while back > cut0 and not is_turn_start(rest[back]):
                    back -= 1
                cut = back if back > cut0 else cut0
        return cut, trim_until

    def _summary_lines(self, rest, fps, start, end, previous: str) -> list[str]:
        lines = [f"previous summary: {previous}"] if previous else []
        for i in range(start, end):
            m = self._trimmed(rest[i], fps[i])  # summarizer gets the short tool outputs
            lines.append(f"{m.get('role', '')}: {message_text(m)}")
        return lines

    def _compress(self, messages: list[Message], summarize_fn) -> list[Message]:
        with self._lock:
            msgs = list(messages)
            n_system = 0
            while n_system < len(msgs) and msgs[n_system].get("role") == "system":
                n_system += 1
            system, rest = msgs[:n_system], msgs[n_system:]
            sys_fps = [fingerprint(m) for m in system]
            fps = [fingerprint(m) for m in rest]
            sizes = [self._size(m, f) for m, f in zip(rest, fps)]
            system_size = sum(self._size(m, f) for m, f in zip(system, sys_fps))
            system_fp = hashlib.blake2b("".join(sys_fps).encode(), digest_size=8).hexdigest()
            before = system_size + sum(sizes)

            # Saved state is only valid for the same system prompt and an unchanged prefix.
            if self._active and (
                system_fp != self._system_fp
                or len(fps) < len(self._hashes)
                or fps[: len(self._hashes)] != self._hashes
            ):
                self._clear_state()

            if self._active:
                out = self._build(system, rest, fps, self._cut, self._trim_until, self._summary, system_size)
                after = self._out_tokens
                if after <= self.max_tokens:  # stable prefix: only new messages appended
                    self._maybe_prefetch(system_size, rest, fps, sizes, after, summarize_fn)
                    self.last_stats = Stats(before, after, self._cut, bool(self._summary), self._n_trimmed, False)
                    return out
            elif before <= self.max_tokens:
                self._maybe_prefetch(system_size, rest, fps, sizes, before, summarize_fn)
                self.last_stats = Stats(before, before, 0, False)
                return msgs

            # ---- compaction event: the prefix changes once, then stays stable again
            was_active = self._active
            cut0, trim0, old_summary = self._cut, self._trim_until, self._summary
            cut, trim_until = self._plan(system_size, rest, fps, sizes, cut0, self._trim_until)
            summary = old_summary
            if summarize_fn is not None:
                ready = self._take_prefetch(cut0, old_summary, rest, fps, sizes, system_size, trim_until)
                if ready is not None:
                    cut, summary = ready
                elif cut > cut0:
                    summary = self._cap_summary(summarize_fn(self._summary_lines(rest, fps, cut0, cut, old_summary)))
            self._pending = None
            self._active = True
            self._system_fp = system_fp
            self._hashes = fps
            self._cut, self._trim_until, self._summary = cut, trim_until, summary
            out = self._build(system, rest, fps, cut, trim_until, summary, system_size)
            changed = not was_active or (cut, trim_until, summary) != (cut0, trim0, old_summary)
            self.last_stats = Stats(before, self._out_tokens, cut, bool(summary), self._n_trimmed, changed)
            return out

    # ---- background summaries

    def _maybe_prefetch(self, system_size, rest, fps, sizes, current: int, summarize_fn) -> None:
        if not (self.background and summarize_fn is not None and self._pending is None):
            return
        if current < self.max_tokens * self.prefetch_ratio:
            return
        cut, trim_until = self._plan(system_size, rest, fps, sizes, self._cut, self._trim_until)
        if cut <= self._cut:
            return
        lines = self._summary_lines(rest, fps, self._cut, cut, self._summary)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="delbina")
        future = self._executor.submit(summarize_fn, lines)
        self._pending = (self._cut, self._summary, fps[:cut], cut, future)

    def _take_prefetch(self, cut0, old_summary, rest, fps, sizes, system_size, trim_until):
        """(cut, summary) prepared in the background, if it still fits this chat."""
        if self._pending is None:
            return None
        p_cut0, p_summary, p_hashes, p_cut, future = self._pending
        if p_cut0 != cut0 or p_summary != old_summary or fps[: len(p_hashes)] != p_hashes:
            future.cancel()
            return None
        # Accept only if it leaves real room to grow; otherwise compact properly now.
        kept = sum(self._eff_size(rest, fps, sizes, i, trim_until) for i in range(p_cut, len(rest)))
        if system_size + self.summary_tokens + kept > self.max_tokens * (1 + self.target_ratio) / 2:
            future.cancel()
            return None
        try:
            return p_cut, self._cap_summary(future.result())
        except Exception:
            return None


def compress(messages: list[Message], **options: Any) -> list[Message]:
    """One-shot compression: fits the chat just under ``max_tokens``.

    A one-shot call has no memory of earlier turns, so it cannot keep the prefix
    stable for prompt caching. For multi-turn chats and agents use one
    ``Compressor`` per conversation instead. Options are Compressor's arguments
    (``target_ratio`` defaults to 1.0 here).
    """
    options.setdefault("target_ratio", 1.0)
    return Compressor(**options).compress(messages)
