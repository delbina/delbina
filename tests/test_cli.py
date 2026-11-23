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


def test_compress_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("delbina.tokens.count", lambda s: len(s.split()))
    f = tmp_path / "chat.json"
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": "w " * 20} for i in range(20)]
    f.write_text(json.dumps({"messages": msgs}))
    out = tmp_path / "out.json"
    assert main(["compress", str(f), "--max-tokens", "100", "--keep-last", "2", "-o", str(out)]) == 0
    assert len(json.loads(out.read_text())["messages"]) < 20


def test_cost_command(tmp_path, capsys):
    f = tmp_path / "chat.json"
    msgs = [{"role": "system", "content": "s"}]
    for i in range(30):
        msgs += [{"role": "user", "content": "q " * 200}, {"role": "assistant", "content": "a " * 200}]
    f.write_text(json.dumps(msgs))
    assert main(["cost", str(f), "--max-tokens", "3000", "--tokenizer", "chars", "--min-prefix", "100"]) == 0
    out = capsys.readouterr().out
    assert "sliding window" in out and "delbina" in out


def test_eval_offline_command(capsys):
    assert main(["eval", "--offline", "--cases", "3", "--steps", "10", "--max-tokens", "800", "--tokenizer", "chars"]) == 0
    assert "overall" in capsys.readouterr().out
