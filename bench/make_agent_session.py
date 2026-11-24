# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Generate bench/data/agent_session.json: a synthetic coding-agent session.

A user gives an agent 25 tasks; the agent reads files, greps and runs tests.
Tool outputs (file contents, search results, test logs) dominate the tokens,
as they do in real agents. Deterministic (seeded), so results are reproducible.

    python bench/make_agent_session.py
"""
import json
import random
from pathlib import Path

rng = random.Random(7)
MODULES = ["auth", "billing", "cache", "db", "api", "jobs", "search", "users", "mailer", "reports"]
WORDS = "value result config client token session request handler payload retry index buffer item record".split()


def code_file(mod: str, n: int) -> str:
    lines = [f'"""{mod} module."""', "import logging", "from typing import Any", "", "log = logging.getLogger(__name__)", ""]
    for i in range(n):
        name = f"{rng.choice(WORDS)}_{rng.choice(WORDS)}_{i}"
        a, b = rng.sample(WORDS, 2)
        lines += [
            f"def {name}({a}: Any, {b}: int = {rng.randint(0, 99)}) -> dict:",
            f'    """Handle {a} for {mod}, up to {b} times."""',
            f"    if not {a}:",
            f'        log.warning("{mod}: empty {a}")',
            "        return {}",
            f"    out = {{'{a}': {a}, 'n': {b} * {rng.randint(2, 9)}}}",
            "    return out",
            "",
        ]
    return "\n".join(lines)


def grep_output(term: str, n: int) -> str:
    return "\n".join(
        f"src/{rng.choice(MODULES)}/{rng.choice(WORDS)}.py:{rng.randint(1, 400)}:    {term} = {rng.choice(WORDS)}({rng.choice(WORDS)})"
        for _ in range(n)
    )


def test_log(mod: str, n: int, fail: bool) -> str:
    lines = ["============================= test session starts =============================",
             f"collected {n} items", ""]
    for i in range(n):
        lines.append(f"tests/test_{mod}.py::test_{rng.choice(WORDS)}_{i} PASSED{' ' * 20}[{int(100 * (i + 1) / n):3d}%]")
    if fail:
        lines += ["", "=================================== FAILURES ===================================",
                  f"___________________ test_{mod}_edge ___________________",
                  f"    assert {mod}_result['n'] == 42", "E   AssertionError: assert 41 == 42",
                  f"=========================== 1 failed, {n - 1} passed in 2.31s ==========================="]
    else:
        lines.append(f"============================== {n} passed in 1.87s ==============================")
    return "\n".join(lines)


def call(i: int, name: str, args: dict) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def main() -> None:
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools read_file, grep and run_tests. "
                                           "Be concise. The repository is a Python web service."}]
    k = 0
    for t in range(25):
        mod = rng.choice(MODULES)
        msgs.append({"role": "user", "content": f"Task {t + 1}: fix the failing edge case in the {mod} module and make sure the tests pass."})
        steps = [("grep", {"pattern": f"{mod}_result"}), ("read_file", {"path": f"src/{mod}/core.py"}),
                 ("run_tests", {"path": f"tests/test_{mod}.py"})]
        if rng.random() < 0.5:
            steps.append(("read_file", {"path": f"src/{mod}/util.py"}))
        steps.append(("run_tests", {"path": f"tests/test_{mod}.py"}))
        for s, (name, args) in enumerate(steps):
            k += 1
            msgs.append(call(k, name, args))
            if name == "grep":
                out = grep_output(f"{mod}_result", rng.randint(15, 60))
            elif name == "read_file":
                out = code_file(mod, rng.randint(15, 45))
            else:
                out = test_log(mod, rng.randint(10, 40), fail=s < len(steps) - 1)
            msgs.append({"role": "tool", "tool_call_id": f"call_{k}", "content": out})
        msgs.append({"role": "assistant", "content": f"Fixed the off-by-one in {mod}_result; all {mod} tests pass now."})
    path = Path(__file__).parent / "data" / "agent_session.json"
    path.write_text(json.dumps({"messages": msgs}, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(msgs)} messages)")


if __name__ == "__main__":
    main()
