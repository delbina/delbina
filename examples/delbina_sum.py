# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Summarize a short agent chat with delbina-sum through Ollama, and check what it kept.

    ollama pull hf.co/delbina/delbina-sum-GGUF:Q4_K_M
    python examples/delbina_sum.py
    python examples/delbina_sum.py --model delbina-sum      # a local `ollama create` name
"""
from __future__ import annotations

import argparse
import time

from delbina import DELBINA_SUM, checked, ollama_summarizer

CHAT = [
    "user: We're moving the billing service (project 'ledger') to the new cluster. Target cluster is prod-blr-3.",
    "assistant: Got it. I'll check the current config for ledger first.",
    "tool: $ cat ledger/deploy.yaml\nreplicas: 3\nport: 8443\nimage: ledger-api:4.12.0\nregion: ap-south-1",
    "user: Use port 9443 instead of 8443, security asked for it. Also the DB password is [[SECRET_1]].",
    "assistant: Updated ledger to port 9443. Should I bump the image too?",
    "user: Maybe 4.13.0? Actually no, keep 4.12.0, 4.13 breaks the invoice export.",
    "user: Separate thing: for project 'atlas', the on-call owner is Priya and the alert threshold is 3%.",
    "assistant: Noted. The atlas dashboard is at grafana/d/atlas-errors.",
    "tool: CI run 8812 failed: test_invoice_rounding timed out after 30s",
    "user: Ignore that flaky test for now, we'll fix it after the move.",
]
# value -> project it must sit under (None: only check that the value is kept)
EXPECT = {"prod-blr-3": "ledger", "9443": "ledger", "ledger-api:4.12.0": "ledger", "[[SECRET_1]]": "ledger",
          "Priya": "atlas", "3%": "atlas", "grafana/d/atlas-errors": "atlas"}
STALE = ["8443", "4.13.0"]  # replaced or rejected: should not be listed as facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DELBINA_SUM)
    ap.add_argument("--host", default="http://localhost:11434")
    a = ap.parse_args()

    dropped: list[str] = []
    summarize = checked(ollama_summarizer(a.model, host=a.host), on_drop=dropped.extend)
    t = time.time()
    summary = summarize(CHAT)
    print(f"--- summary ({time.time() - t:.1f}s) ---\n{summary}\n")

    facts = summary.split("Context:")[0]
    for value, project in EXPECT.items():
        lines = [l for l in facts.splitlines() if value in l]
        ok = bool(lines) and (project is None or any(project in l for l in lines))
        print(f"{'OK  ' if ok else 'MISS'} {value}" + ("" if ok else f"  (expected under {project})"))
    for value in STALE:
        hit = any(value in l for l in facts.splitlines())
        print(f"{'STALE' if hit else 'OK   '} {value} {'listed as a fact' if hit else 'not listed'}")
    print(f"fact check dropped: {dropped or 'nothing'}")


if __name__ == "__main__":
    main()
