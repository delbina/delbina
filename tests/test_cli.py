import json

from delbina import ollama_summarizer
from delbina.cli import main


def test_bench_runs(tmp_path, capsys):
    f = tmp_path / "chat.json"
    f.write_text(json.dumps({"turns": [f"turn {i} " + "word " * 10 for i in range(8)]}))
    assert main(["bench", str(f)]) == 0
    out = capsys.readouterr().out
    assert "full" in out and "last_4" in out


def test_ollama_summarizer_parses_response(monkeypatch):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps({"response": " short summary "}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Resp())
    assert ollama_summarizer()(["a", "b"]) == "short summary"
