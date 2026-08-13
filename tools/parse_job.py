"""Job description parser — port of the n8n `jd_data` half of 'analysisPrompt'.

Haiku 4.5 structured extraction with hard minimum counts enforced in code
(minimums come from the n8n prompt: >= 6 required skills, >= 6 key
responsibilities, >= 5 verbatim key phrases, >= 3 outcomes).

The parsed `JobData` drives every downstream stage: gap analysis decides
severity from the missing skills/phrases, generation must cover each one
(mandatory JD coverage, bug #5), and the keyword-cap guardrail counts mentions
of `required_skills` + `key_phrases` (bug #4).
"""
from __future__ import annotations

from typing import Callable

import config
from schemas import JobData
from tools.llm import structured_call

TokenCallback = Callable[[str], None]

JD_ANALYSIS_SYSTEM = """\
You are a precise job-description parser. Extract every requirement and desired \
signal from a job posting so a resume can be tailored to it later. Quote the \
posting; never invent requirements that are not present.

Extract:
- role_title: the exact job title from the posting.
- summary_of_role: a 1-2 sentence plain-language summary of what the role does \
and the team/context around it.
- required_skills: a list of AT LEAST 6 distinct skills or technologies the \
posting requires (e.g. "Python", "Kubernetes", "stakeholder management"). \
Extract as many as exist — do not stop at 6.
- key_responsibilities: a list of AT LEAST 6 distinct responsibilities. Write \
them as concise verbatim-ish phrases drawn from the posting.
- preferred_qualifications: any "nice to have" / preferred qualifications.
- key_phrases: a list of AT LEAST 5 VERBATIM short phrases taken word-for-word \
from the posting that capture its priorities (e.g. "high-performance systems", \
"scale across multiple regions"). These must be exact quotes — punctuation \
included where possible.
- stakeholders: the people/teams the role interacts with (e.g. "engineering", \
"product", "C-suite", "external clients"), if stated.
- outcomes: a list of AT LEAST 3 business outcomes the role is expected to \
drive (e.g. "reduce time-to-market", "improve reliability", "grow revenue").

Rules:
- Every required skill / responsibility / phrase must trace to the actual \
posting text. Do not add generic skills the posting never mentions.
- Do not collapse distinct skills into one item.
- If the posting is short, still return at least the stated minimums from what \
is present; never fabricate to pad counts.
"""


def validate_jd_counts(jd: JobData) -> list[str]:
    """Deterministic minimum-count check (bug #5 groundwork).

    Returns a list of human-readable warnings; empty list == all minimums met.
    These are logged/streamed by the agent — never silently swallowed.
    """
    warnings: list[str] = []
    if len(jd.required_skills) < config.MIN_REQUIRED_SKILLS:
        warnings.append(
            f"required_skills has {len(jd.required_skills)} items "
            f"(min {config.MIN_REQUIRED_SKILLS})"
        )
    if len(jd.key_responsibilities) < config.MIN_KEY_RESPONSIBILITIES:
        warnings.append(
            f"key_responsibilities has {len(jd.key_responsibilities)} items "
            f"(min {config.MIN_KEY_RESPONSIBILITIES})"
        )
    if len(jd.key_phrases) < config.MIN_KEY_PHRASES:
        warnings.append(
            f"key_phrases has {len(jd.key_phrases)} items "
            f"(min {config.MIN_KEY_PHRASES})"
        )
    if len(jd.outcomes) < config.MIN_OUTCOMES:
        warnings.append(
            f"outcomes has {len(jd.outcomes)} items "
            f"(min {config.MIN_OUTCOMES})"
        )
    return warnings


def parse_job_description(
    jd_text: str,
    *,
    on_token: TokenCallback | None = None,
) -> JobData:
    """Extract structured `JobData` from a job-description text."""
    user = (
        "Extract the job requirements from the following posting. Follow every "
        "rule in the system prompt exactly.\n\n--- JOB DESCRIPTION ---\n\n"
        + jd_text.strip()
    )
    return structured_call(
        model=config.MODEL_PARSE,
        system=JD_ANALYSIS_SYSTEM,
        user=user,
        output_model=JobData,
        max_tokens=4000,
        on_token=on_token,
    )
