"""Validation & repair pass — port of the n8n 'Validation Prompt'.

A second, cheaper LLM pass (Haiku 4.5) that scans the generated resume against
the ORIGINAL resume text and the JD, and emits a CORRECTED ResumeContent in the
same schema — never a prose report.

The critical design point (bug #5): the generated resume is intentionally
allowed to contain every JD requirement, even ones the candidate's original
resume does not show. Validation is told the exact list of deliberately-added
content (`intentionally_added`) so it never strips JD coverage as if it were a
hallucination.
"""
from __future__ import annotations

import json
from typing import Callable

import config
from schemas import JobData, ResumeContent
from tools.llm import structured_call

TokenCallback = Callable[[str], None]

VALIDATION_SYSTEM = """\
You are a strict resume validator. You compare a GENERATED resume against the \
ORIGINAL resume text and a JOB DESCRIPTION, then return the CORRECTED resume.

Run all NINE checks:

1. FAKE CLIENT EXISTENCE — every named client under an employer must appear in \
ORIGINAL_RESUME_TEXT with real work described for it. A client name that is \
not in ORIGINAL_RESUME_TEXT is fabricated: remove it, merge its bullets into \
the employer-level bullets, and set has_clients false if it was the only \
client. If a real name appears but the described work is not in the original, \
rewrite those bullets from what the original actually says.
2. CLIENT ATTRIBUTION — a bullet that cannot be attributed to a specific \
client (or mixes several) must be moved up to the employer level or rewritten \
generically. No client may claim another client's work.
3. DOMAIN-BLENDED SENTENCES — if one sentence mixes two unrelated client \
domains (e.g. a banking fact welded to an airline fact), split it into two \
separate, standalone bullets, one per domain/client.
4. REPEATED DESCRIPTIONS — no two clients (or employers) may share the same \
or near-identical description/bullets. Rewrite duplicates to be distinct.
5. CONTACT COMPLETENESS — every contact field (name, phone, email, linkedin) \
that exists in ORIGINAL_RESUME_TEXT must be present in the contact object. \
Never invent a field that is not in the original.
6. SECTION COMPLETENESS — do not drop sections that exist in the original \
resume. Preserve employer names, dates, and order.
7. JD COVERAGE (ATS) — every required skill, responsibility, and key phrase \
from JOB_DESCRIPTION_DATA must appear at least once in the resume. If one is \
missing, ADD ONE natural bullet at the single most relevant employer. This is \
MANDATORY, not optional. If any keyword appears more than 3 times total, \
reduce the excess.
8. KEYWORD NATURALNESS — keywords must be embedded in complete sentences with \
an action and, where the original states one, an outcome. Never leave a bare \
keyword list. If a skill exists only in the skills section, make sure it also \
appears naturally in at least one sentence.
9. DUPLICATE CLIENT DETECTION — a client name appearing under more than one \
employer is an error. Keep it only under the employer where the work is \
actually described; remove the copies from the other employers.

CRITICAL — READ TWICE:
This resume is INTENTIONALLY designed to include every required skill and \
responsibility from JOB_DESCRIPTION_DATA, even ones the candidate's original \
resume doesn't show. That is correct, expected behavior — NOT a hallucination. \
Do NOT remove, flag, or 'correct' content just because it isn't found in \
ORIGINAL_RESUME_TEXT. The section titled INTENTIONALLY_ADDED lists exactly the \
JD content that was added on purpose — leave all of it in place unless a \
check above requires a structural fix.

Output ONLY the corrected resume JSON in the exact same schema as \
GENERATED_RESUME_JSON. Do not add commentary, markdown, or fields outside the \
schema.
"""


def validate_and_repair(
    original_resume_text: str,
    jd_data: JobData,
    generated_content: ResumeContent,
    *,
    intentionally_added: list[str] | None = None,
    on_token: TokenCallback | None = None,
) -> ResumeContent:
    """Second LLM pass: correct the generated resume in place.

    `intentionally_added` = the JD skills/phrases the gap analysis found missing
    from the source resume (bug #5) — validation must not strip them.
    """
    user = "\n".join(
        [
            "Run all nine checks, then return the corrected resume.",
            "",
            "## ORIGINAL_RESUME_TEXT (ground truth — the only factual source)",
            original_resume_text.strip(),
            "",
            "## JOB_DESCRIPTION_DATA",
            json.dumps(jd_data.model_dump(), indent=1),
            "",
            "## GENERATED_RESUME_JSON (to validate and correct)",
            json.dumps(generated_content.model_dump(), indent=1),
            "",
            "## INTENTIONALLY_ADDED (JD content deliberately added — do NOT strip)",
            json.dumps(intentionally_added or [], indent=1),
        ]
    )
    # Combo gateways AUTO-think even when thinking isn't requested, and a
    # full-resume correction burns a lot of budget. On the direct API the small
    # budget is plenty; on a gateway we start at GATEWAY_START_TOKENS (64k) —
    # the measured happy-path budget where think+emit fit in one attempt. If it
    # burns out on thinking alone, tools/llm.py recovers once at up to
    # GATEWAY_MAX_TOKENS, and every attempt is wall-clock-bounded there too.
    max_tokens = (
        config.GATEWAY_START_TOKENS if config.is_gateway() else config.DIRECT_MAX_TOKENS_VALIDATE
    )
    return structured_call(
        model=config.MODEL_VALIDATE,
        system=VALIDATION_SYSTEM,
        user=user,
        output_model=ResumeContent,
        max_tokens=max_tokens,
        on_token=on_token,
    )
