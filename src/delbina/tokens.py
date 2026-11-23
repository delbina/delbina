from functools import lru_cache

import tiktoken


@lru_cache(maxsize=None)
def _encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    # Loaded lazily: tiktoken downloads the file on first use, so importing
    # delbina must not need network.
    return tiktoken.get_encoding(name)


def count(text: str) -> int:
    return len(_encoding().encode(text))


def cumulative_cost(turns: list[str]) -> int:
    """Total tokens if the full history is re-sent each turn."""
    total, history = 0, 0
    for t in turns:
        history += count(t)
        total += history
    return total
