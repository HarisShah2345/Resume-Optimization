"""Deterministic alignment-gap analysis.

The agent calls this AFTER parsing to decide the tailoring strategy:
- `full_rebuild`   — the resume covers little of the JD; rewrite substantially.
- `heavy_tailoring`— meaningful gaps to add/reframe.
- `light_tweak`    — strong fit; minor polish only.

Severity is computed in code from JD coverage (required skills + verbatim
phrases present in the ORIGINAL resume text). No LLM judgment call — the model
never gets to self-report its own fit.

Note: the gap function needs the raw resume text (not just the parsed
structure) because keyword presence must be checked against ground truth.
"""
from __future__ import annotations

import re

import config
from schemas import GapAnalysis, JobData, ResumeStructure
from tools.guardrails import count_keyword_mentions

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
    "into", "across", "that", "this", "their", "from", "by", "at", "as",
    "our", "we", "its", "them", "they", "be", "is", "are", "was", "were",
    "will", "has", "have", "had", "more", "most", "than", "over", "under",
}


def _content_words(phrase: str) -> list[str]:
    """Meaningful tokens of a phrase (len >= 4, not a stopword)."""
    words = re.findall(r"[a-z][a-z0-9+-]*", phrase.lower())
    return [w for w in words if len(w) >= 4 and w not in _STOPWORDS]


def _missing_outcomes(resume_text: str, jd_data: JobData) -> list[str]:
    """An outcome is 'missing' when none of its content words appear anywhere
    in the original resume — i.e. no supporting evidence exists to reframe."""
    text = resume_text.lower()
    missing = []
    for outcome in jd_data.outcomes:
        words = _content_words(outcome)
        if words and not any(w in text for w in words):
            missing.append(outcome)
    return missing


def check_alignment_gap(
    resume_text: str,
    resume_structure: ResumeStructure,
    jd_data: JobData,
) -> GapAnalysis:
    """Compute gap severity + exactly which JD content is missing."""
    text = resume_text.lower()

    missing_skills = [
        s for s in jd_data.required_skills if count_keyword_mentions(text, s) == 0
    ]
    missing_phrases = [p for p in jd_data.key_phrases if p.lower() not in text]
    missing_outcomes = _missing_outcomes(text, jd_data)

    total = max(1, len(jd_data.required_skills))
    covered = total - len(missing_skills)
    coverage = covered / total

    if coverage < 0.4 or len(missing_phrases) >= 4:
        severity = "full_rebuild"
    elif coverage < 0.75:
        severity = "heavy_tailoring"
    else:
        severity = "light_tweak"

    depth_target = resume_structure.estimated_seniority
    reasoning = (
        f"Resume covers {covered}/{total} required JD skills "
        f"({coverage:.0%}). Missing skills: "
        f"{', '.join(missing_skills) or 'none'}. Missing verbatim phrases: "
        f"{len(missing_phrases)}. Severity: {severity} (depth target: "
        f"{depth_target})."
    )

    return GapAnalysis(
        severity=severity,
        missing_skills=missing_skills,
        missing_phrases=missing_phrases,
        missing_outcomes=missing_outcomes,
        depth_target=depth_target,
        reasoning=reasoning,
    )
