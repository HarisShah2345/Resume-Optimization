"""Resume structure parser — port of the n8n 'analysisPrompt' + 'Fetching
Output1' + grounding guardrails.

Two layers:
  1. LLM extraction (Haiku 4.5, structured output) of the resume's shape:
     sections_present, employers, named clients with per-client bullet counts,
     seniority estimate, and which contact fields exist.
  2. Deterministic client-detection safety check (the hardest-won fix): the
     LLM can PROPOSE client blocks, but code enforces the >= MIN_BULLETS
     floor and drops anything that isn't grounded in the original text.

`sections_present` is the sole authority for employer set and section order —
the generation stage must match it 1:1 (bug #2).
"""
from __future__ import annotations

from typing import Callable

import config
from schemas import ResumeStructure
from tools.guardrails import filter_clients_by_min_bullets
from tools.llm import structured_call

TokenCallback = Callable[[str], None]

ANALYSIS_SYSTEM = """\
You are a precise resume parser. You extract a resume's structure so it can be \
tailored to a job description later. You do not rewrite, judge, or embellish \
anything — you report what is actually present.

Extract the following:

sections_present — the section headings in the resume, in order of appearance, \
as the resume spells them (e.g. ["PROFESSIONAL SUMMARY", "WORK EXPERIENCE", \
"SKILLS", "CERTIFICATIONS", "EDUCATION"]). Include every section you can see.

employers — one entry per employer (company), in order of appearance. For each:
- name: exactly as it appears in the resume.
- start_date / end_date: if present (e.g. "2020" or "Jan 2020").
- clients: named clients, customers, or consulting engagements this employer \
served (only for consultancies / agencies / service firms — NOT employers that \
did their own product work). Only include a client if it is a REAL, specifically \
named company/segment with genuine work described. Generic phrases \
("various clients", "global partners") and names mentioned in passing with no \
work described must NOT be listed.
- bullet_count for each client: the number of resume sentences that are \
individually attributed to THAT client and describe real work done for them. \
CRITICAL — a collective mention that lists several partners at once \
("partners including X, Y and Z") attributes work to no single client: count 0 \
for each. Only sentences that specifically name one client and describe \
achievement for it count toward that client's bullet_count.
- has_clients: TRUE ONLY if at least 2 DISTINCT named clients each have at \
least 2 individually-attributed achievement sentences. Otherwise FALSE.
- Do NOT invent employers, dates, or clients that are not in the resume.

employer_count — total number of employers extracted.

has_projects_section / has_publications_section — whether the resume contains \
a PROJECTS or PUBLICATIONS section.

estimated_seniority — estimate from the depth and seniority of the roles: \
"entry" (early career, ~1 page), "mid" (established, ~2 pages), or "senior" \
(10+ years, leadership, complex scope, ~3 pages).

contact_fields_found — which of these are present in the resume: name, phone, \
email, linkedin. Use exactly these lowercase tokens.

Rules:
- Every employer in the resume must appear; do not merge, split, or rename.
- Do not add any client with fewer than 2 individually-attributed sentences — \
the downstream filter will reject them anyway.
- Be exhaustive but do not fabricate.
"""


def parse_resume_structure(
    resume_text: str,
    *,
    on_token: TokenCallback | None = None,
) -> ResumeStructure:
    """Extract and sanitize the resume structure.

    The Haiku extraction is post-processed by the deterministic
    `filter_clients_by_min_bullets` guardrail (bug #1), so a fabricated client
    can never survive into the generated resume.
    """
    user = (
        "Extract the structure of the following resume. Follow every rule in "
        "the system prompt exactly.\n\n--- RESUME ---\n\n"
        + resume_text.strip()
    )
    structure = structured_call(
        model=config.MODEL_PARSE,
        system=ANALYSIS_SYSTEM,
        user=user,
        output_model=ResumeStructure,
        max_tokens=4000,
        on_token=on_token,
    )

    # Hard guarantee (bug #1): drop clients with < MIN_BULLETS_PER_CLIENT.
    structure.employers = filter_clients_by_min_bullets(structure.employers)
    structure.employer_count = len(structure.employers)
    return structure
