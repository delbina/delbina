# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Best-effort secret redaction for text sent to a summarizer.

Modes:
- ``"mask"``: secrets become ``[[SECRET_n]]`` before the summarizer sees them and
  are put back into the summary afterwards. The value never leaves this process,
  but the summary still holds it. The mapping lives in memory only.
- ``"strip"``: secrets become ``[REDACTED:kind]`` for good. Use this for anything
  you store or publish (datasets, logs).

This is pattern matching, not a guarantee: unknown token formats and secrets
written in plain words are not caught.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Iterable, Pattern

MODES = ("mask", "strip")

# (kind, regex). If the regex has a group named "v", only that part is replaced.
PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("private_key", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+(?P<v>[A-Za-z0-9._~+/\-]{20,}=*)")),
    ("url_password", re.compile(r"(?<=://)[^/\s:@]+:(?P<v>[^/\s@]{3,})@")),
    ("assigned_secret", re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)"
        r"\b[\"']?\s*(?:[:=]|\s+is\s+)\s*([\"']?)(?P<v>[^\s\"',;]{6,})\1")),
]

_PLACEHOLDER = re.compile(r"\[\[SECRET_\d+\]\]|\[REDACTED:[a-z_]+\]")
_RESTORE = re.compile(r"\[{1,2}SECRET_(\d+)\]{1,2}")
_CANDIDATE = re.compile(r"[A-Za-z0-9+/_\-=]{24,}")


def _entropy(s: str) -> float:
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


class Redactor:
    """Finds secrets in text and masks or strips them. Thread-safe.

    Args:
        mode: ``"mask"`` (reversible, in memory) or ``"strip"`` (one-way).
        patterns: ``(kind, compiled regex)`` pairs; default ``PATTERNS``.
        entropy: also catch long random-looking tokens (off by default: it
            also hits hashes and UUID-like IDs).
    """

    def __init__(self, mode: str = "mask", patterns: Iterable[tuple[str, Pattern[str]]] | None = None,
                 entropy: bool = False, min_entropy: float = 4.0) -> None:
        if mode not in MODES:
            raise ValueError(f"redact mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.patterns = list(PATTERNS if patterns is None else patterns)
        self.entropy = entropy
        self.min_entropy = min_entropy
        self.counts: Counter[str] = Counter()   # kinds found, cumulative (never values)
        self._lock = threading.Lock()
        self._by_value: dict[str, str] = {}
        self._by_key: dict[str, str] = {}

    def find(self, text: str) -> list[tuple[int, int, str]]:
        """Non-overlapping (start, end, kind) spans of secrets in ``text``."""
        spans = []
        for kind, rx in self.patterns:
            for m in rx.finditer(text):
                if "v" in rx.groupindex and m.group("v") is not None:
                    spans.append((m.start("v"), m.end("v"), kind))
                else:
                    spans.append((m.start(), m.end(), kind))
        if self.entropy:
            for m in _CANDIDATE.finditer(text):
                s = m.group()
                if any(c.isdigit() for c in s) and any(c.isalpha() for c in s) and _entropy(s) >= self.min_entropy:
                    spans.append((m.start(), m.end(), "high_entropy"))
        spans = [sp for sp in spans if not _PLACEHOLDER.fullmatch(text[sp[0]:sp[1]])]
        spans.sort(key=lambda sp: (sp[0], -(sp[1] - sp[0])))
        out, last = [], -1
        for sp in spans:
            if sp[0] >= last:
                out.append(sp)
                last = sp[1]
        return out

    def redact(self, text: str, counts: Counter[str] | None = None) -> str:
        """Replace secrets in ``text``. Kinds found are added to ``self.counts`` and to ``counts``."""
        spans = self.find(text)
        if not spans:
            return text
        parts, pos = [], 0
        with self._lock:
            for start, end, kind in spans:
                parts.append(text[pos:start])
                parts.append(self._placeholder(text[start:end], kind))
                self.counts[kind] += 1
                if counts is not None:
                    counts[kind] += 1
                pos = end
        parts.append(text[pos:])
        return "".join(parts)

    def restore(self, text: str) -> str:
        """Put masked values back (mask mode). Unknown placeholders are left as they are."""
        if self.mode != "mask" or not text:
            return text
        with self._lock:
            return _RESTORE.sub(lambda m: self._by_key.get(m.group(1), m.group(0)), text)

    def _placeholder(self, value: str, kind: str) -> str:
        if self.mode == "strip":
            return f"[REDACTED:{kind}]"
        key = self._by_value.get(value)
        if key is None:
            key = str(len(self._by_value) + 1)
            self._by_value[value] = key
            self._by_key[key] = value
        return f"[[SECRET_{key}]]"


def redact_text(text: str, mode: str = "strip", **options) -> str:
    """One-shot redaction. Default ``strip`` (nothing to restore from)."""
    return Redactor(mode, **options).redact(text)


def resolve(redact) -> Redactor | None:
    if redact is None or redact is False:
        return None
    if redact is True:
        return Redactor("mask")
    if isinstance(redact, str):
        return Redactor(redact)
    if isinstance(redact, Redactor):
        return redact
    raise TypeError("redact must be None, True, 'mask', 'strip' or a Redactor")
