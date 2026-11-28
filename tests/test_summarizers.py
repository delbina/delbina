import io
import json

import delbina.ollama as ollama
from delbina import DELBINA_SUM, ollama_summarizer, openai_summarizer


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture(monkeypatch, reply):
    bodies = []

    def urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _Resp(json.dumps(reply).encode())

    monkeypatch.setattr(ollama.urllib.request, "urlopen", urlopen)
    return bodies


def test_ollama_defaults_match_delbina_sum_eval(monkeypatch):
    bodies = _capture(monkeypatch, {"response": "Facts:\n- a: b: c"})
    ollama_summarizer(DELBINA_SUM)(["x", "y"])
    opts = bodies[0]["options"]
    assert bodies[0]["model"] == DELBINA_SUM
    assert opts["temperature"] == 0.0 and opts["num_ctx"] == 8192 and opts["num_predict"] == 400


def test_ollama_options_override(monkeypatch):
    bodies = _capture(monkeypatch, {"response": "ok"})
    ollama_summarizer(temperature=0.7, num_ctx=4096, max_tokens=200)(["x"])
    assert bodies[0]["options"] == {"num_predict": 200, "temperature": 0.7, "num_ctx": 4096}


def test_prompt_unchanged_for_delbina_sum():
    # delbina-sum was trained on exactly this prompt (delbina-sum imports PROMPT from here)
    assert ollama._prompt(["a", "b"], 400) == ollama.PROMPT.format(chat="- a\n- b")


def test_openai_temperature_only_when_set(monkeypatch):
    reply = {"choices": [{"message": {"content": "s"}}]}
    bodies = _capture(monkeypatch, reply)
    openai_summarizer(api_key="k")(["x"])
    openai_summarizer(api_key="k", temperature=0)(["x"])
    assert "temperature" not in bodies[0]
    assert bodies[1]["temperature"] == 0
