# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
from delbina import FactCheck, check_facts, checked

CHAT = ["user: final call on the kestrel storage: postgres",
        "tool: deploy/merlin.env:3: HOST=merlin-stg-07.internal",
        "user: Pin httpx to 3.32.11 in falcon, newer releases break the build.",
        "user: the api key is [[SECRET_1]]"]


def test_keeps_copied_values_drops_invented():
    summ = ("Facts:\n"
            "- kestrel: storage backend: postgres\n"
            "- kestrel: env var: POSTGRES_USER\n"
            "- merlin: staging host: `merlin-stg-07.internal`.\n"
            "- falcon: rate limit: 1000 (newer releases break the build)\n"
            "- falcon: httpx version: 3.32.11\n"
            "- falcon: base branch: master\n"
            "- falcon: API key: [[SECRET_1]]\n"
            "Context:\n"
            "- falcon: newer httpx releases break the build.")
    r = check_facts(summ, CHAT)
    assert isinstance(r, FactCheck)
    assert r.dropped == ["- kestrel: env var: POSTGRES_USER",
                         "- falcon: rate limit: 1000 (newer releases break the build)",
                         "- falcon: base branch: master"]
    assert "postgres" in r.text and "merlin-stg-07.internal" in r.text and "[[SECRET_1]]" in r.text
    assert r.text.endswith("- falcon: newer httpx releases break the build.")  # Context untouched


def test_case_insensitive_and_string_source():
    r = check_facts("Facts:\n- kestrel: storage: Postgres", "\n".join(CHAT))
    assert r.dropped == [] and r.text == "Facts:\n- kestrel: storage: Postgres"


def test_context_lines_are_not_checked():
    summ = "Facts:\n- a: b: postgres\nContext:\n- x: y: totally made up"
    assert check_facts(summ, CHAT).dropped == []


def test_empty_facts_header_removed():
    r = check_facts("Facts:\n- a: b: nope\nContext:\n- a: note", CHAT)
    assert r.text == "Context:\n- a: note"
    assert check_facts("Facts:\n- a: b: nope", CHAT).text == ""


def test_non_fact_text_passes_through():
    prose = "The team picked postgres. Nothing else was decided."
    assert check_facts(prose, CHAT).text == prose


def test_checked_wrapper():
    seen = []
    summarize = checked(lambda turns: "Facts:\n- a: b: postgres\n- a: c: master", on_drop=seen.append)
    assert summarize(CHAT) == "Facts:\n- a: b: postgres"
    assert seen == [["- a: c: master"]]
    assert summarize.__name__ == "checked_<lambda>"
