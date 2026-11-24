# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Compare token cost of compression strategies on a chat file."""
import json
import sys
from pathlib import Path

from delbina import cost, count, full, head_tail, last_n, rolling_summary

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "sample_chat.json"
turns = json.loads(path.read_text(encoding="utf-8"))["turns"]

strategies = [full, last_n(4), head_tail(1, 3), rolling_summary(keep=2)]

base = cost(turns, full)
print(f"{path.name}: {len(turns)} turns, {sum(count(t) for t in turns)} tokens once\n")
print(f"{'strategy':<16}{'tokens':>8}{'saved':>8}")
for s in strategies:
    c = cost(turns, s)
    print(f"{s.__name__:<16}{c:>8}{1 - c / base:>8.0%}")
