from delbina.evaluate import make_cases, report, run

words = lambda s: len(s.split())


def test_offline_eval_runs():
    cases = make_cases(n=3, steps=12, seed=3)
    assert all(c.answer in str(c.messages) for c in cases)
    scores = run(cases, max_tokens=600, ask=None, count_tokens=words)
    by = {s.mode: s for s in scores}
    assert by["full"].accuracy == 1.0
    assert by["delbina"].tokens < by["full"].tokens
    assert "overall" in report(scores, offline=True)


def test_summary_mode_with_model_stub():
    cases = make_cases(n=3, steps=12, seed=3)
    # A "perfect" summarizer keeps every user line; a stub model echoes its context.
    summarize = lambda lines: " ".join(l for l in lines if l.startswith(("user:", "previous summary:")))
    ask = lambda ctx: "\n".join(str(m.get("content")) for m in ctx)
    scores = run(cases, max_tokens=600, ask=ask, summarize=summarize, count_tokens=words)
    assert {s.mode: s for s in scores}["summary"].accuracy == 1.0
