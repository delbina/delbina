"""If tiktoken can't download its encoding (offline machine), fall back to a
word-based fake so the suite still runs. CI has network and uses the real one."""
import pytest

import delbina.tokens as tokens


class _WordEncoding:
    def encode(self, text, **_):
        return text.split()


@pytest.fixture(autouse=True, scope="session")
def _offline_tokenizer():
    try:
        tokens._encoding("cl100k_base")
    except Exception:
        tokens._encoding.cache_clear()
        original = tokens._encoding
        tokens._encoding = lambda name="cl100k_base": _WordEncoding()
        yield
        tokens._encoding = original
        return
    yield
