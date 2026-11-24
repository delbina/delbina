# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""A tool-calling agent loop with delbina, runnable offline.

The "model" here is a stub that reads a few files and then answers, so you can
see what delbina does without an API key: watch tool outputs being trimmed,
compaction events, and the state file that survives restarts.

    python examples/agent_loop.py
    python examples/agent_loop.py    # run again: history + delbina state restored from agent_state.json

Swap `fake_llm` for a real client call (OpenAI, Anthropic, Ollama ...).
"""
import json
from pathlib import Path

from delbina import Compressor

STATE = Path("agent_state.json")


def read_file(path: str) -> str:
    return "\n".join(f"{path}:{i}: some_code_here(value_{i})" for i in range(300))


def fake_llm(messages: list[dict], step: int) -> dict:
    """Stand-in for a model: calls read_file three times per task, then answers."""
    if step % 4 < 3:
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{step}", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/mod_{step}.py"})}}]}
    return {"role": "assistant", "content": f"Done with task {step // 4}."}


def main() -> None:
    comp = Compressor(max_tokens=12000, keep_last=6, tool_max_tokens=150, keep_tool_results=2, count_tokens="chars")
    history = [{"role": "system", "content": "You are a coding agent."}]
    step = first_task = 0
    if STATE.exists():  # a real agent persists its history; delbina's state goes next to it
        saved = json.loads(STATE.read_text())
        history, step, first_task = saved["history"], saved["step"], saved["next_task"]
        comp.load_state(saved["delbina"])
        print(f"restored {len(history)} messages, summary/cut state intact\n")
    for task in range(first_task, first_task + 4):
        history.append({"role": "user", "content": f"Task {task}: review three files."})
        while True:
            msg = fake_llm(comp.compress(history), step)
            s = comp.last_stats
            print(f"step {step:>2}: {s.tokens_before:>6} -> {s.tokens_after:>5} tokens"
                  f"{'  COMPACTED' if s.compacted else ''}{f', {s.trimmed} tool outputs trimmed' if s.trimmed else ''}")
            history.append(msg)
            step += 1
            if not msg.get("tool_calls"):
                break
            for call in msg["tool_calls"]:
                args = json.loads(call["function"]["arguments"])
                history.append({"role": "tool", "tool_call_id": call["id"], "content": read_file(args["path"])})
    STATE.write_text(json.dumps({"history": history, "step": step, "next_task": first_task + 4,
                                 "delbina": comp.state()}))
    print(f"\nsaved to {STATE}; run again to continue")


if __name__ == "__main__":
    main()
