# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Verbatim check for summaries in the "Facts: / Context:" format (delbina-sum's output).

A small summarizer model can write a fact line whose value was never in the chat:
a made-up default ("base branch: master") or a name built from nearby words
("env var: POSTGRES_USER"). The next model then treats it as true. ``check_facts``
drops every fact line whose value does not occur in the source text, so a summary
can lose a fact but never gain one.

    from delbina import checked, ollama_summarizer
    summarize = checked(ollama_summarizer(...))     # same signature, filtered output

The check is a case-insensitive substring match against the text the summarizer saw
(after redaction, so "[[SECRET_1]]" placeholders pass). A value with words glued on
("8080 (was 7070)") is dropped too: it is no longer a clean copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Summarizer = Callable[[list[str]], str]


@dataclass
class FactCheck:
    text: str                                         # the summary with bad fact lines removed
    dropped: list[str] = field(default_factory=list)  # the removed lines, as written


def _value(line: str) -> str | None:
    """Value of a "- subject: label: value" line, else None (not a fact line)."""
    s = line.strip()
    if not s.startswith(("- ", "* ")):
        return None
    parts = s[2:].split(": ", 2)
    if len(parts) != 3:
        return None
    return parts[2].strip().rstrip(".;,").strip("`").strip() or None


def check_facts(summary: str, source: str | list[str]) -> FactCheck:
    """Drop fact lines (before a "Context:" line) whose value is not in ``source``."""
    text = (source if isinstance(source, str) else "\n".join(source)).lower()
    out: list[str] = []
    dropped: list[str] = []
    in_facts = True
    for line in summary.splitlines():
        if line.strip() == "Context:":
            in_facts = False
        v = _value(line) if in_facts else None
        if v is not None and v.lower() not in text:
            dropped.append(line)
            continue
        out.append(line)
    # a "Facts:" header with nothing left under it goes too
    kept = [i for i, l in enumerate(out) if l.strip() == "Facts:"]
    for i in reversed(kept):
        nxt = out[i + 1].strip() if i + 1 < len(out) else ""
        if nxt in ("", "Context:"):
            del out[i]
    return FactCheck("\n".join(out).strip("\n"), dropped)


def checked(summarize: Summarizer, on_drop: Callable[[list[str]], None] | None = None) -> Summarizer:
    """Wrap a summarizer so its output passes ``check_facts`` against its own input."""
    def run(turns: list[str]) -> str:
        res = check_facts(summarize(turns), turns)
        if res.dropped and on_drop:
            on_drop(res.dropped)
        return res.text
    run.__name__ = f"checked_{getattr(summarize, '__name__', 'summarizer')}"
    return run
