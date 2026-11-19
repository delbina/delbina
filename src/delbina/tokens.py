import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    return len(_enc.encode(text))


def cumulative_cost(turns: list[str]) -> int:
    """Total tokens if the full history is re-sent each turn."""
    total, history = 0, 0
    for t in turns:
        history += count(t)
        total += history
    return total
