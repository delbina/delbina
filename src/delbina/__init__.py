# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""delbina: compress chat context before it reaches the model.

    from delbina import Compressor
    comp = Compressor(max_tokens=8000)          # one per conversation
    small = comp.compress(messages)             # cache-friendly, trims old tool output

Or run it as a proxy in front of any OpenAI-compatible API:

    delbina serve --upstream https://api.openai.com/v1 --max-tokens 8000
"""
__version__ = "0.4.0"

from .strategies import cost, full, head_tail, last_n, naive_summarizer, rolling_summary
from .messages import BudgetError, CompactionEvent, Compressor, Stats, compress, message_text
from .ollama import ollama_summarizer, openai_summarizer
from .factcheck import FactCheck, check_facts, checked
from .redact import Redactor, redact_text
from .tokens import count, counter, cumulative_cost
from .trim import is_tool_result, trim_tool_message

__all__ = [
    "__version__",
    # main API
    "compress", "Compressor", "Stats", "CompactionEvent", "BudgetError", "message_text",
    "ollama_summarizer", "openai_summarizer",
    "check_facts", "checked", "FactCheck",
    # privacy
    "Redactor", "redact_text",
    # tool outputs
    "is_tool_result", "trim_tool_message",
    # token measurement
    "count", "counter", "cumulative_cost", "cost",
    # turn-list strategies (benchmarking)
    "full", "last_n", "head_tail", "rolling_summary", "naive_summarizer",
]
