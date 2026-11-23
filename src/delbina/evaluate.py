"""Answer-quality benchmark: does the model still answer correctly after compression?

Each case is a long agent-style conversation (chat turns plus large tool
outputs) with one planted fact, e.g. "the deploy key for project orion is
K7Q2ZP", placed early, in the middle, or near the end. The final user message
asks for that fact. Every compression mode is scored on the same cases:

- ``full``       the whole conversation (reference; may exceed small models' context)
- ``sliding``    sliding window, drop only (delbina 0.2 behaviour)
- ``delbina``    cache-friendly compaction + tool trimming, drop only
- ``summary``    same, plus a running summary (needs --summary-model)

Scoring is exact: the answer is correct if the planted value appears in the
model's reply. ``--offline`` skips the model and reports whether the value is
still present in the compressed context at all (an upper bound for accuracy).

    delbina eval --base-url http://localhost:11434/v1 --model qwen2.5:7b --summary-model qwen2.5:1.5b
    delbina eval --offline
    delbina eval --base-url https://generativelanguage.googleapis.com/v1beta/openai \\
        --model gemini-3.5-flash-lite --summary-model gemini-3.5-flash-lite \\
        --api-key $GEMINI_API_KEY --reasoning-effort minimal --rpm 12 --cases 12
"""
from __future__ import annotations

import json
import os
import random
import string
from dataclasses import dataclass, field
from typing import Callable

from .messages import Compressor, message_text
from .ollama import RateLimiter, _post, finish_reason, openai_summarizer, reply_text
from .tokens import TokenCounter, resolve

PROJECTS = ["orion", "lyra", "vega", "draco", "hydra", "cygnus", "aquila", "carina", "pavo", "lupus"]
FACT_TEMPLATES = [
    ("The deploy key for project {p} is {v}. Keep it handy.", "What is the deploy key for project {p}? Reply with the key only."),
    ("Note for later: the {p} service listens on port {v}.", "Which port does the {p} service listen on? Reply with the number only."),
    ("The on-call engineer for {p} this week has the badge ID {v}.", "What is the badge ID of the {p} on-call engineer? Reply with the ID only."),
]
POSITIONS = {"early": 0.1, "middle": 0.5, "late": 0.85}
MODES = ("full", "sliding", "delbina", "summary")


@dataclass
class Case:
    messages: list[dict]
    question: str
    answer: str
    position: str


def _value(rng: random.Random, template: str) -> str:
    if "port" in template:
        return str(rng.randint(20000, 65000))
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))


def _filler(rng: random.Random, i: int) -> list[dict]:
    """One agent step: user request, tool call, tool output, short reply."""
    mod = rng.choice(PROJECTS)
    lines = "\n".join(
        f"src/{mod}/{rng.choice(['core', 'util', 'api'])}.py:{rng.randint(1, 900)}: "
        f"{rng.choice(['def', 'return', 'raise', 'log'])} {''.join(rng.choices(string.ascii_lowercase, k=8))}"
        for _ in range(rng.randint(20, 80))
    )
    return [
        {"role": "user", "content": f"Step {i}: check the {mod} module for unused helpers."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "grep", "arguments": json.dumps({"q": mod})}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": lines},
        {"role": "assistant", "content": f"Checked {mod}: {rng.randint(0, 5)} unused helpers found, nothing else changed."},
    ]


def make_cases(n: int = 30, steps: int = 40, seed: int = 1) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    positions = list(POSITIONS)
    for k in range(n):
        position = positions[k % len(positions)]
        fact_t, question_t = FACT_TEMPLATES[k % len(FACT_TEMPLATES)]
        project = rng.choice(PROJECTS)
        value = _value(rng, fact_t)
        at = int(steps * POSITIONS[position])
        msgs = [{"role": "system", "content": "You are a helpful coding agent. Answer briefly."}]
        for i in range(steps):
            if i == at:
                msgs.append({"role": "user", "content": fact_t.format(p=project, v=value)})
                msgs.append({"role": "assistant", "content": "Noted."})
            msgs.extend(_filler(rng, i))
        cases.append(Case(msgs, question_t.format(p=project), value, position))
    return cases


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    timeout: float = 300,
    reasoning_effort: str | None = None,
    retries: int = 0,
    throttle: Callable[[], None] | None = None,
    max_tokens: int = 50,
    on_empty: Callable[[str], None] | None = None,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> str:
    body: dict = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _post(base_url.rstrip("/") + "/chat/completions", body, headers, timeout, retries, throttle, on_retry)
    text = reply_text(data)
    if not text and on_empty is not None:
        on_empty(finish_reason(data))
    return text


def parse_modes(text: str | None, have_summary: bool) -> list[str]:
    """``"full,delbina"`` -> ["full", "delbina"]. None = every mode available."""
    if not text:
        return [m for m in MODES if m != "summary" or have_summary]
    modes = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f"unknown mode(s): {', '.join(bad)} (choose from {', '.join(MODES)})")
    if "summary" in modes and not have_summary:
        raise ValueError("mode 'summary' needs --summary-model")
    return modes


@dataclass
class Score:
    mode: str
    correct: dict[str, int] = field(default_factory=dict)
    total: dict[str, int] = field(default_factory=dict)
    tokens: int = 0
    empty: int = 0
    errors: int = 0

    def add(self, position: str, ok: bool, tokens: int, empty: bool = False) -> None:
        self.empty += empty
        self.correct[position] = self.correct.get(position, 0) + ok
        self.total[position] = self.total.get(position, 0) + 1
        self.tokens += tokens

    @property
    def accuracy(self) -> float:
        t = sum(self.total.values())
        return sum(self.correct.values()) / t if t else 0.0


def run(
    cases: list[Case],
    max_tokens: int,
    ask: Callable[[list[dict]], str] | None,
    summarize=None,
    count_tokens: TokenCounter | str | None = None,
    progress: Callable[[str], None] | None = None,
    modes: list[str] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> list[Score]:
    counter = resolve(count_tokens)
    common = dict(max_tokens=max_tokens, keep_last=4, count_tokens=counter)
    all_modes: list[tuple[str, Callable[[], Compressor] | None]] = [
        ("full", None),
        ("sliding", lambda: Compressor(**common, target_ratio=1.0, tool_max_tokens=None)),
        ("delbina", lambda: Compressor(**common)),
    ]
    if summarize is not None:
        all_modes.append(("summary", lambda: Compressor(**common, summarize=summarize)))
    selected = [(n, f) for n, f in all_modes if modes is None or n in modes]
    scores = []
    for name, make in selected:
        score = Score(name)
        for i, case in enumerate(cases):
            if progress:
                progress(f"{name} case {i + 1}/{len(cases)}")
            try:
                # Replay turn by turn so stateful modes see the conversation grow, as in real use.
                comp = make() if make else None
                if comp is not None:
                    for end in range(2, len(case.messages) + 1):
                        if case.messages[end - 1].get("role") in ("user", "tool"):
                            comp.compress(case.messages[:end])
                msgs = case.messages + [{"role": "user", "content": case.question}]
                ctx = comp.compress(msgs) if comp else msgs
                tokens = sum(counter(message_text(m)) for m in ctx)
                if ask is None:
                    ok = case.answer in "\n".join(message_text(m) for m in ctx[:-1])
                    empty = False
                else:
                    reply = ask(ctx)
                    ok, empty = case.answer in reply, not reply
            except Exception as e:
                if ask is None and summarize is None:
                    raise  # offline: no network involved, so this is a real bug
                score.errors += 1  # network/API failure after all retries: skip the case, keep going
                if on_error is not None:
                    on_error(f"{name} case {i + 1}: {type(e).__name__}: {e}")
                continue
            score.add(case.position, ok, tokens, empty)
        scores.append(score)
    return scores


def report(scores: list[Score], offline: bool) -> str:
    positions = list(POSITIONS)
    head = "fact still in context" if offline else "answered correctly"
    out = [f"{head} (by where the fact was in the chat)\n",
           f"{'mode':<10}{'overall':>9}" + "".join(f"{p:>9}" for p in positions) + f"{'avg tokens':>12}"]
    show_empty = any(s.empty for s in scores)
    show_err = any(s.errors for s in scores)
    if show_empty:
        out[-1] += f"{'no reply':>10}"
    if show_err:
        out[-1] += f"{'errors':>8}"
    for s in scores:
        n = sum(s.total.values()) or 1
        row = f"{s.mode:<10}{s.accuracy:>9.0%}"
        for p in positions:
            row += f"{(s.correct.get(p, 0) / s.total[p]) if s.total.get(p) else 0:>9.0%}"
        row += f"{s.tokens // n:>12,}"
        out.append(row + (f"{s.empty:>10}" if show_empty else "") + (f"{s.errors:>8}" if show_err else ""))
    if show_err:
        out.append("\n'errors' = API/network failure after all retries; those cases are left out of the scores.")
    if show_empty:
        out.append("\n'no reply' = the model returned no text; counted as wrong.")
    return "\n".join(out)


def main_eval(a) -> None:
    import sys
    cases = make_cases(a.cases, a.steps, a.seed)
    ask = None
    summarize = None
    calls = {"n": 0, "label": ""}
    if not a.offline:
        key = a.api_key if a.api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        throttle = RateLimiter(a.rpm)  # one limiter shared by answers and summaries
        on_retry = lambda n, wait, why: print(
            f"\n  {why}, retry {n}/{a.retries} in {wait:.0f}s", file=sys.stderr, flush=True)
        opts = dict(reasoning_effort=a.reasoning_effort, retries=a.retries, throttle=throttle,
                    on_retry=on_retry, timeout=a.timeout)
        reasons: dict[str, int] = {}
        on_empty = lambda r: reasons.__setitem__(r, reasons.get(r, 0) + 1)
        ask = lambda msgs: chat_completion(a.base_url, a.model, msgs, key, on_empty=on_empty, **opts)
        if a.summary_model:
            raw = openai_summarizer(a.summary_model, base_url=a.base_url, api_key=key, **opts)

            def summarize(lines):  # show each summary call, so slow runs don't look stuck
                calls["n"] += 1
                print(f"\r{calls['label']}, summary call {calls['n']}   ", end="", file=sys.stderr, flush=True)
                return raw(lines)
    elif a.summary_model:
        print("note: --summary-model is ignored with --offline", file=sys.stderr)
    try:
        modes = parse_modes(a.modes, summarize is not None)
    except ValueError as e:
        sys.exit(f"delbina eval: {e}")
    def progress(label: str) -> None:
        calls["label"] = label
        print(f"\r{label}" + " " * 24, end="", file=sys.stderr, flush=True)

    on_error = lambda msg: print(f"\n  skipped {msg}", file=sys.stderr, flush=True)
    scores = run(cases, a.max_tokens, ask, summarize, a.tokenizer,
                 None if a.offline else progress, modes, on_error)
    if not a.offline:
        print(file=sys.stderr)
    print(f"{len(cases)} cases x {a.steps} agent steps, budget {a.max_tokens} tokens"
          + ("" if a.offline else f", model {a.model}") + "\n")
    print(report(scores, a.offline))
    if not a.offline and reasons:
        print("no-reply finish reasons: " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
