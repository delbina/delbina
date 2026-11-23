"""delbina: compress chat context before it reaches the model.

    from delbina import compress
    messages = compress(messages, max_tokens=2000)
"""
__version__ = "0.2.0"

from .strategies import cost, full, head_tail, last_n, naive_summarizer, rolling_summary
from .messages import Compressor, Stats, compress, message_text
from .ollama import ollama_summarizer
from .tokens import count, cumulative_cost

__all__ = [
    "__version__",
    # main API
    "compress", "Compressor", "Stats", "message_text", "ollama_summarizer",
    # token measurement
    "count", "cumulative_cost", "cost",
    # turn-list strategies (benchmarking)
    "full", "last_n", "head_tail", "rolling_summary", "naive_summarizer",
]
