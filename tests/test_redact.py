"""Secret redaction: the summarizer never sees secrets; mask mode keeps them in the summary."""
import pytest

from delbina import Compressor, Redactor, redact_text

words = lambda s: len(s.split())

SECRETS = {
    "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
    "github_token": "ghp_" + "a1B2" * 9,
    "google_api_key": "AIza" + "Sy" + "x" * 33,
    "openai_key": "sk-proj-" + "Ab3" * 10,
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
}


@pytest.mark.parametrize("kind,value", SECRETS.items())
def test_detects_known_formats(kind, value):
    out = redact_text(f"use {value} now")
    assert value not in out and f"[REDACTED:{kind}]" in out


def test_assignment_url_pem_bearer():
    text = ('password = "hunter22x" and https://bob:s3cretpw@db.local/x\n'
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123\n"
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----")
    out = redact_text(text)
    for v in ("hunter22x", "s3cretpw", "abcdefghijklmnopqrstuvwxyz123", "MIIEow"):
        assert v not in out
    assert "password" in out and "bob" in out and "db.local" in out   # context kept


def test_no_false_positives_on_plain_text():
    text = "port 8443, file src/app/main.py, commit 3f2a9c1, uuid 123e4567-e89b-12d3-a456-426614174000"
    assert redact_text(text) == text


def test_mask_round_trip_is_stable():
    r = Redactor("mask")
    v = SECRETS["aws_access_key"]
    a, b = r.redact(f"key {v}"), r.redact(f"again {v}")
    assert a == "key [[SECRET_1]]" and b == "again [[SECRET_1]]"
    assert r.restore("the key is [SECRET_1]") == f"the key is {v}"     # tolerant of one bracket
    assert r.counts["aws_access_key"] == 2


def test_entropy_is_opt_in():
    tok = "Zx9Qk2Lp7Rt4Vw8Ny3Mb6Hc1Fd5Gs0Jq"
    assert redact_text(f"token {tok}") == f"token {tok}"
    assert tok not in redact_text(f"token {tok}", entropy=True)


def chat_with_secret(value, n=30):
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": f"the deploy key is {value} keep it"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": "a " * 30}, {"role": "user", "content": f"u{i} " * 30}]
    return msgs


@pytest.mark.parametrize("background", [False, True])
def test_mask_hides_from_summarizer_but_summary_keeps_it(background):
    value = SECRETS["github_token"]
    sent = []

    def summarize(lines):
        sent.extend(lines)
        return "key: " + next(w for l in lines for w in l.split() if "SECRET_" in w)

    comp = Compressor(max_tokens=300, count_tokens=words, summarize=summarize, summary_tokens=20,
                      redact=True, background=background)
    msgs = chat_with_secret(value)
    for end in range(2, len(msgs) + 1):
        out = comp.compress(msgs[:end])
    comp.close()
    assert sent and not any(value in l for l in sent)
    assert value in comp.summary and value in str(out[1]["content"])


def test_strip_never_restores():
    value = SECRETS["aws_access_key"]
    comp = Compressor(max_tokens=300, count_tokens=words, summary_tokens=20, redact="strip",
                      summarize=lambda lines: " ".join(w for l in lines for w in l.split() if "REDACTED" in w))
    comp.compress(chat_with_secret(value))
    assert value not in comp.summary and "[REDACTED:aws_access_key]" in comp.summary


def test_bad_mode():
    with pytest.raises(ValueError):
        Compressor(redact="blur")
