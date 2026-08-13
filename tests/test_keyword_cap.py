"""Tests for the keyword-frequency discipline (bug #4) — counted in code,
never an LLM self-report. A JD keyword may appear at most 3 times total."""
from __future__ import annotations

from schemas import (
    ContentClient,
    ContentContact,
    ContentEmployer,
    JobData,
    ResumeContent,
    SkillCategory,
)
from tools.generate_content import run_post_generation_checks
from tools.guardrails import (
    collect_content_text,
    count_keyword_mentions,
    enforce_keyword_cap,
    find_over_limit_keywords,
)

JD = JobData(
    role_title="Data Engineer",
    summary_of_role="Pipelines.",
    required_skills=["Python", "Kubernetes", "Kafka"],
    key_responsibilities=["a", "b", "c", "d", "e", "f"],
    key_phrases=["data quality", "streaming pipelines"],
    outcomes=["reliability"],
)


def _content(**kw) -> ResumeContent:
    base = ResumeContent(
        contact=ContentContact(name="J", email="j@c.com", phone="1", linkedin="l"),
        summary="Engineer in Python.",
        employers=[
            ContentEmployer(
                name="Co",
                role_title="Engineer",
                has_clients=False,
                bullets=[
                    "Used Python daily.",
                    "Python for data.",
                    "Built Python services.",
                    "Streaming pipelines in Python.",
                ],
            )
        ],
        skills=[SkillCategory(category="Core", items=["Python", "Kafka"])],
    )
    return base.model_copy(update=kw)


def test_counts_across_all_content_structures():
    text = collect_content_text(_content())
    # 1 summary + 4 bullets + 1 skills item = 6 mentions of Python.
    assert count_keyword_mentions(text, "Python") == 6


def test_phrase_with_spaces_counted_as_whole():
    content = _content(
        summary="Streaming pipelines data quality and streaming pipelines again.",
        employers=[
            ContentEmployer(
                name="Co",
                role_title="Engineer",
                has_clients=False,
                bullets=["Kubernetes clusters and Kafka topics."],
            )
        ],
    )
    text = collect_content_text(content)
    assert count_keyword_mentions(text, "streaming pipelines") == 2


def test_over_limit_flags_only_repeat_keywords():
    content = _content()
    over = find_over_limit_keywords(content, JD, max_freq=3)
    assert ("Python", 6) in over
    assert not any(kw == "Kubernetes" for kw, _ in over)  # never used
    assert not any(kw == "Kafka" for kw, _ in over)  # used once


def test_within_limit_passes():
    content = _content()
    content.employers[0].bullets = [
        "Used Python daily.",
        "Streaming pipelines in Python.",
        "Kubernetes clusters.",
    ]
    content.summary = "Engineer."
    content.skills = [SkillCategory(category="Core", items=["Python", "Kafka", "Kubernetes"])]
    over = find_over_limit_keywords(content, JD, max_freq=3)
    assert over == []


def test_postcheck_emits_keyword_cap_issue_with_repair():
    content = _content()  # Python x6: 1 summary + 4 bullets + 1 skills tag
    fixed, report = run_post_generation_checks(content, JD, "Python Co", "mid")
    # The redundant skills-tag mention is auto-trimmed (5 mentions live only
    # in bullet prose, which the auto-fix never touches), so the reported
    # count reflects what's left after that fix, not the original 6.
    assert "Python" not in fixed.skills[0].items
    cap = [i for i in report.issues if i.code == "keyword_cap"]
    assert cap
    assert "5" in cap[0].message
    assert "at most 3" in cap[0].repair


def test_case_insensitive_counting():
    text = "PYTHON and python and Python"
    assert count_keyword_mentions(text, "python") == 3


def test_no_false_match_inside_words():
    text = "Kubernetes orchestrates; pythonic style"
    assert count_keyword_mentions(text, "python") == 0
    assert count_keyword_mentions(text, "Kubernetes") == 1


def test_enforce_keyword_cap_trims_redundant_skills_tag():
    content = _content()  # Python x6: 1 summary + 4 bullets + 1 skills tag
    fixed, still_over = enforce_keyword_cap(content, JD, max_freq=3)
    assert "Python" not in fixed.skills[0].items
    assert "Kafka" in fixed.skills[0].items  # untouched — Kafka is within cap
    assert ("Python", 5) in still_over  # 5 live only in prose, needs a repair instruction
    # Bullet text is never touched by the auto-fix.
    assert fixed.employers[0].bullets == content.employers[0].bullets


def test_enforce_keyword_cap_trims_tools_lists_too():
    content = _content(
        employers=[
            ContentEmployer(
                name="Co",
                role_title="Engineer",
                has_clients=False,
                bullets=["Used Python daily.", "Python for data."],
                tools=["Python", "Kafka"],
            )
        ],
        skills=[SkillCategory(category="Core", items=["Python"])],
    )
    # Python: 1 summary + 2 bullets + 1 tools tag + 1 skills tag = 5.
    fixed, still_over = enforce_keyword_cap(content, JD, max_freq=3)
    total = count_keyword_mentions(collect_content_text(fixed), "Python")
    assert total == 3
    assert still_over == []


def test_enforce_keyword_cap_leaves_within_limit_untouched():
    content = _content()
    content.employers[0].bullets = ["Used Python daily.", "Kubernetes clusters."]
    content.summary = "Engineer."
    fixed, still_over = enforce_keyword_cap(content, JD, max_freq=3)
    assert still_over == []
    assert fixed.skills[0].items == content.skills[0].items  # nothing needed trimming


def test_client_bullets_included_in_count():
    content = _content()
    content.employers[0].has_clients = True
    content.employers[0].bullets = []
    content.employers[0].clients = [
        ContentClient(name="Bank", bullets=["Python banking"], tools=["Kafka"]),
        ContentClient(name="Air", bullets=["Python analytics", "more Python"], tools=["Python"]),
    ]
    text = collect_content_text(content)
    # summary 1 + client bullets/tools 4 + skills item 1 = 6
    assert count_keyword_mentions(text, "Python") == 6
