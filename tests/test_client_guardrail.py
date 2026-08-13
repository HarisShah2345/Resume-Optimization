"""Tests for the deterministic client-detection safety check (bug #1).

This was the single hardest-won fix in the original n8n build: the LLM is free
to PROPOSE client blocks, but code enforces the >= MIN_BULLETS floor and drops
anything not grounded in the original resume text.
"""
from __future__ import annotations

from schemas import (
    ClientProbe,
    ContentClient,
    ContentEmployer,
    EmployerProbe,
    ResumeContent,
)
from tools.guardrails import (
    count_keyword_mentions,
    enforce_client_grounding,
    filter_clients_by_min_bullets,
    is_grounded,
)

RESUME = """
Professional Summary: Senior consultant with 12 years of experience.

Work Experience:
Acme Consulting — 2018 to Present
  Nimbus Bank: Migrated their core ledger to microservices. Cut monthly
  reconciliation time by 60%. Achieved 99.99% uptime.
  AeroFly: Built their booking API serving 50K requests a day. Introduced
  feature flags that enabled zero-downtime deploys.
  Led a 10-person delivery team.
"""


# ---------------------------------------------------------------------------
# is_grounded — whole-word, case-insensitive presence in the source text
# ---------------------------------------------------------------------------
def test_grounded_exact_name():
    assert is_grounded("Nimbus Bank", RESUME)


def test_grounded_case_and_whitespace_insensitive():
    assert is_grounded("  nimbus bank  ", RESUME)


def test_grounded_fails_when_name_absent():
    assert not is_grounded("Totally Fake Corp", RESUME)


def test_grounded_requires_word_boundary_not_substring():
    # "ai" must not match inside "achieved" / "AeroFly".
    assert not is_grounded("ai", RESUME)
    assert not is_grounded("Aero", RESUME)


def test_grounded_rejects_blank_and_one_char_names():
    assert not is_grounded("  ", RESUME)
    assert not is_grounded("A", RESUME)


def test_grounded_tolerates_pdf_extraction_whitespace_noise():
    """`pypdf.extract_text()` on a real uploaded resume routinely turns the
    single space between two words into a newline (or extra spaces) when
    they wrapped across a visual line or column boundary — verified live.
    A multi-word client name is exactly the kind of text that wraps, so an
    exact-space match flattened EVERY employer's real clients on real
    uploaded resumes: a false positive on the fabrication guardrail, not an
    actual fabrication. Whitespace between the name's words must match any
    run of whitespace in the source, not just a literal single space."""
    assert is_grounded("Nimbus Bank", "Nimbus\nBank: migrated their ledger.")
    assert is_grounded("Nimbus Bank", "Nimbus  Bank: migrated their ledger.")
    assert is_grounded("Nimbus Bank", "Nimbus \n  Bank: migrated their ledger.")
    # Still correctly rejects words that are genuinely joined with no
    # separator at all — this isn't "match anything containing these letters".
    assert not is_grounded("Nimbus Bank", "NimbusBank: migrated their ledger.")


# ---------------------------------------------------------------------------
# filter_clients_by_min_bullets — port of 'Fetching Output1' (n8n code node)
# ---------------------------------------------------------------------------
def test_keeps_clients_with_two_or_more_bullets():
    emp = EmployerProbe(
        name="Acme Consulting",
        has_clients=True,
        clients=[
            ClientProbe(name="Nimbus Bank", bullet_count=3),
            ClientProbe(name="AeroFly", bullet_count=2),
        ],
    )
    cleaned = filter_clients_by_min_bullets([emp])
    assert cleaned[0].has_clients is True
    assert [c.name for c in cleaned[0].clients] == ["Nimbus Bank", "AeroFly"]


def test_drops_single_bullet_client():
    emp = EmployerProbe(
        name="Acme Consulting",
        has_clients=True,
        clients=[
            ClientProbe(name="Nimbus Bank", bullet_count=3),
            ClientProbe(name="OneHit Wonder", bullet_count=1),  # < 2 -> dropped
        ],
    )
    cleaned = filter_clients_by_min_bullets([emp])
    assert [c.name for c in cleaned[0].clients] == ["Nimbus Bank"]


def test_flattens_employer_when_no_client_survives():
    # Only collective-mention clients (bullet_count 0) + a 1-bullet client.
    emp = EmployerProbe(
        name="Acme Consulting",
        has_clients=True,
        clients=[
            ClientProbe(name="Collective Partners", bullet_count=0),
            ClientProbe(name="OneHit Wonder", bullet_count=1),
        ],
    )
    cleaned = filter_clients_by_min_bullets([emp])
    assert cleaned[0].has_clients is False
    assert cleaned[0].clients == []


def test_collective_mention_never_counts():
    # A single sentence listing three partners attributes work to no one.
    emp = EmployerProbe(
        name="Acme Consulting",
        has_clients=True,
        clients=[
            ClientProbe(name="Nimbus Bank", bullet_count=0),  # named in passing
            ClientProbe(name="AeroFly", bullet_count=0),
            ClientProbe(name="SkyCorp", bullet_count=0),
        ],
    )
    cleaned = filter_clients_by_min_bullets([emp])
    assert cleaned[0].has_clients is False
    assert cleaned[0].clients == []


def test_employer_without_clients_untouched():
    emp = EmployerProbe(name="Product Co", has_clients=False, clients=[])
    cleaned = filter_clients_by_min_bullets([emp])
    assert cleaned == [emp]


# ---------------------------------------------------------------------------
# enforce_client_grounding — port of 'Fetching output 2' (n8n code node)
# ---------------------------------------------------------------------------
def test_ungrounded_client_flattens_employer_merging_bullets_and_tools():
    emp = ContentEmployer(
        name="Acme Consulting",
        role_title="",
        has_clients=True,
        clients=[
            ContentClient(name="RealClient", bullets=["b1"], tools=["AWS"]),
            ContentClient(name="HallucinatedClient", bullets=["fake bullet"], tools=["Go"]),
        ],
    )
    cleaned = enforce_client_grounding([emp], RESUME)
    flat = cleaned[0]
    assert flat.has_clients is False
    assert flat.clients == []
    assert flat.bullets == ["b1", "fake bullet"]  # client bullets merged up
    assert flat.tools == ["AWS", "Go"]  # client tools merged up


def test_grounded_clients_kept_in_place():
    emp = ContentEmployer(
        name="Acme Consulting",
        role_title="",
        has_clients=True,
        clients=[
            ContentClient(name="Nimbus Bank", bullets=["b1"]),
            ContentClient(name="AeroFly", bullets=["b2"]),
        ],
    )
    cleaned = enforce_client_grounding([emp], RESUME)
    assert cleaned[0].has_clients is True
    assert [c.name for c in cleaned[0].clients] == ["Nimbus Bank", "AeroFly"]


def test_non_client_employer_untouched():
    emp = ContentEmployer(name="Product Co", role_title="", has_clients=False, bullets=["x"])
    cleaned = enforce_client_grounding([emp], RESUME)
    assert cleaned == [emp]


def test_grounded_but_unconfirmed_client_removed_individually():
    """Bug #1's FULL contract: a name grounded in the resume text is necessary
    but not sufficient — the parse must also confirm >= MIN_BULLETS_PER_CLIENT
    attributed sentences. The e2e 'Fabricated MegaCorp' trap is exactly this: a
    single passing mention IS grounded, yet is not a real client and must not
    survive as a promoted client block. A sibling client that IS confirmed
    must survive as a client, not get flattened as collateral damage."""
    emp = ContentEmployer(
        name="Acme Consulting",
        role_title="",
        has_clients=True,
        clients=[
            ContentClient(name="Nimbus Bank", bullets=["Migrated core ledger."]),
            ContentClient(name="AeroFly", bullets=["Built the booking API."]),
        ],
    )
    # Only Nimbus Bank was confirmed by the parse; AeroFly is grounded in the
    # text but was NOT a surviving parsed client.
    cleaned = enforce_client_grounding([emp], RESUME, valid_client_names={"nimbus bank"})
    fixed = cleaned[0]
    assert fixed.has_clients is True
    assert [c.name for c in fixed.clients] == ["Nimbus Bank"]
    # AeroFly's bullet is dropped, not merged up into employer-level bullets
    # (which would be invisible while has_clients stays True).
    assert fixed.bullets == []


def test_fabricated_client_removed_without_flattening_real_sibling():
    """Verified live: a Haiku validation pass renamed a real client
    ('Associated Bank') to a fabricated one ('AWS Glue Migration Project').
    The fabricated one must be dropped, but the untouched real client sitting
    right next to it ('Bank of America') must survive as a client block, not
    get wiped out as collateral damage from the old flatten-the-whole-employer
    behavior."""
    emp = ContentEmployer(
        name="Randstad Technologies",
        role_title="",
        has_clients=True,
        clients=[
            ContentClient(name="Nimbus Bank", bullets=["Migrated core ledger."]),
            ContentClient(name="AWS Glue Migration Project", bullets=["Something fabricated."]),
        ],
    )
    cleaned = enforce_client_grounding([emp], RESUME)
    fixed = cleaned[0]
    assert fixed.has_clients is True
    assert [c.name for c in fixed.clients] == ["Nimbus Bank"]


def test_valid_client_names_keeps_confirmed_clients():
    emp = ContentEmployer(
        name="Acme Consulting",
        role_title="",
        has_clients=True,
        clients=[
            ContentClient(name="Nimbus Bank", bullets=["b1"]),
            ContentClient(name="AeroFly", bullets=["b2"]),
        ],
    )
    cleaned = enforce_client_grounding(
        [emp], RESUME, valid_client_names={"nimbus bank", "aerofly"}
    )
    assert cleaned[0].has_clients is True
    assert [c.name for c in cleaned[0].clients] == ["Nimbus Bank", "AeroFly"]


# ---------------------------------------------------------------------------
# end-to-end: LLM-proposed junk never survives both layers
# ---------------------------------------------------------------------------
def test_two_layer_guardrail_removes_fabrication():
    # Layer 1: 2-bullet filter on the extracted structure.
    structure_emp = EmployerProbe(
        name="Acme Consulting",
        has_clients=True,
        clients=[
            ClientProbe(name="Nimbus Bank", bullet_count=3),
            ClientProbe(name="Fabricated MegaCorp", bullet_count=2),  # passes layer 1
        ],
    )
    filtered = filter_clients_by_min_bullets([structure_emp])

    # Layer 2: grounding check against the ORIGINAL resume text.
    generated = ContentEmployer(
        name="Acme Consulting",
        role_title="",
        has_clients=True,
        clients=[
            ContentClient(name="Nimbus Bank", bullets=["Migrated core ledger."]),
            ContentClient(name="Fabricated MegaCorp", bullets=["Something fake."]),
        ],
    )
    final = enforce_client_grounding([generated], RESUME)
    # Layer 1 keeps both (each has 2 bullets); layer 2 is what catches the
    # fake — Nimbus Bank survives as a client, Fabricated MegaCorp is removed
    # individually rather than taking Nimbus Bank down with it.
    assert final[0].has_clients is True
    assert [c.name for c in final[0].clients] == ["Nimbus Bank"]
    assert len(filtered[0].clients) == 2
    assert not is_grounded("Fabricated MegaCorp", RESUME)


def test_count_keyword_mentions_whole_word():
    text = "Python developer skilled in Python. Pythonic is not a match."
    assert count_keyword_mentions(text, "Python") == 2
    assert count_keyword_mentions(text, "Pythonic") == 1


def test_collect_content_and_count_is_code_not_llm():
    # Keyword counting must be pure code — verify it works on a ResumeContent.
    content = ResumeContent(
        summary="Expert in Python and Python and Python and Python.",
    )
    from tools.guardrails import collect_content_text, find_over_limit_keywords
    from schemas import JobData

    jd = JobData(
        role_title="Python Developer",
        summary_of_role="Write Python.",
        required_skills=["Python"],
        key_responsibilities=[],
        key_phrases=[],
        outcomes=[],
    )
    text = collect_content_text(content)
    assert count_keyword_mentions(text, "Python") == 4
    over = find_over_limit_keywords(content, jd, max_freq=3)
    assert ("Python", 4) in over
