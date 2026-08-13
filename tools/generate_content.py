"""Aligned-content generation — port of the n8n 'aligmentPrompt' + 'Fetching
output 2' post-checks.

The Sonnet 5 call writes the tailored resume (4-phase prompt). THEN a set of
deterministic post-generation checks runs in code — this is the rebuild's
design upgrade over n8n, where the same guarantees were scattered across
prompt lines and code nodes:

  - keyword cap <= 3, counted in code        (bug #4)
  - contact completeness                      (bug #3)
  - duplicate client attribution              (bug #9)
  - client-name grounding against the original resume (bug #1)
  - role-title word cap + uniqueness          (bug #7)
  - seniority-scaled bullet-depth floors      (bug #6)

Safe, unambiguous fixes (grounding flattening) are auto-applied; everything
else becomes a `repair_instruction` the agent feeds into the next pass.
"""
from __future__ import annotations

import json
from typing import Callable

import config
from schemas import (
    GapAnalysis,
    Issue,
    JobData,
    PostCheckReport,
    ResumeContent,
    ResumeStructure,
)
from tools.guardrails import (
    collect_content_text,
    contact_completeness,
    count_keyword_mentions,
    enforce_client_grounding,
    enforce_keyword_cap,
    enforce_single_role_title,
    find_duplicate_clients,
    find_duplicate_role_titles,
    find_overlong_role_titles,
)
from tools.llm import structured_call

TokenCallback = Callable[[str], None]

GENERATION_SYSTEM = """\
You are a senior resume writer who tailors a candidate's existing resume to a \
specific job description. You NEVER invent facts; you reframe, reorder, and \
sharpen what is genuinely present, and add one natural sentence only for JD \
content that is genuinely missing from the source resume.

## PHASE 0 — Absolute constraints (never violate these)
1. Never invent a client or segment sub-header. A client block may only exist \
if it is a real, NAMED client with genuinely described work in the source resume.
2. Never invent, drop, merge, or rename employers. The employer set and section \
order come from `resume_structure.sections_present` — it is the sole authority.
3. Never omit any contact field that exists in the source resume.
4. No keyword may appear more than 3 times across the ENTIRE resume (counted \
later in code — do not try to report your own count).
5. Never invent factual detail: dates, metrics, tools, technologies, \
achievements. Everything must trace to the source resume.
6. Never graft a JD phrase onto a clause it does not belong to. If a JD \
requirement has no matching evidence, add ONE new standalone bullet at the \
most relevant employer — never stretch an existing sentence.
7. When multiple named clients are shown under one employer, their \
descriptions and bullets must be distinct. Never copy-paste the same text.

## PHASE 2 — Output schema
You output a structured resume with: contact, summary, employers (each with \
reframed role_title, description, and either employer bullets OR named \
clients — never both), skills, certifications, education, additional_sections.
- employers[].role_title: ONE natural job title, reframed toward the JD, MAX 6 \
words, unique per employer. Never a compound, comma-separated title like \
"Senior Data Engineer, Cloud ETL" — pick the single title that best captures \
the role.
- employers[].bullets: ONLY when has_clients is false.
- employers[].clients: ONLY when has_clients is true; each client has its own \
bullets (at least 2) describing work done FOR that client.
- Non-core sections (PROJECTS, PUBLICATIONS, LANGUAGES, etc.) go into \
additional_sections, preserving their original title and order.
- Do not include keys the schema does not define.

## PHASE 3 — Priority rules
PRIORITY 1 (highest) — Structural fidelity: match sections_present exactly; \
employers match the parsed list 1:1; never merge/split/rename.
PRIORITY 2 — JD alignment, bounded by the keyword cap: every required skill, \
key responsibility, and key phrase must appear at least once across the \
summary and 2+ employer bullets, and never more than 3 times total. For JD \
content genuinely absent from the source resume, ADD ONE natural bullet at the \
single most relevant employer (mandatory — do not skip it). Reframe \
role_titles and employer descriptions toward the JD.
PRIORITY 3 — Seniority-scaled depth (the depth target is provided per run): \
entry ~1 page (3-5 bullets per role); mid ~2 pages (4-6 per role); senior \
~3 pages (6-8 per role) — senior is a MINIMUM, not a ceiling. The bullet \
requirement applies to an employer's total work including its clients.
PRIORITY 4 — Style: action verbs, varied sentence openings, quantified where \
the source actually states a number.

## PHASE 4 — Silent self-check
Before finishing, verify: (1) no client invented; (2) no employer \
invented/dropped/merged/renamed; (3) every contact field preserved; (4) every \
required skill/responsibility/phrase present at least once and nothing \
repeated more than 3 times; (5) no factual detail invented; (6) the depth \
target is met. Fix anything you find before outputting.
"""


# ---------------------------------------------------------------------------
# PHASE 1 — dynamic input context
# ---------------------------------------------------------------------------
def build_generation_user(
    resume_structure: ResumeStructure,
    jd_data: JobData,
    gap_analysis: GapAnalysis,
    *,
    original_resume_text: str,
    target_title: str | None = None,
    user_emphasis: str | None = None,
    repair_instructions: list[str] | None = None,
) -> str:
    lines: list[str] = [
        "Tailor the candidate's resume to the job description below. "
        "Follow every PHASE rule in the system prompt.",
        "",
        "## resume_structure (parsed from the source resume)",
        json.dumps(resume_structure.model_dump(), indent=1),
        "",
        "## jd_data (parsed from the job description)",
        json.dumps(jd_data.model_dump(), indent=1),
        "",
        "## gap_analysis (deterministic — trust it)",
        json.dumps(gap_analysis.model_dump(), indent=1),
        "",
        f"## desired_job_title\n{target_title or jd_data.role_title}",
        "",
        "## candidate_resume_text (the ONLY source of factual detail)",
        original_resume_text.strip(),
    ]

    if user_emphasis:
        lines += [
            "",
            "## candidate_emphasis_request",
            (
                f"The candidate asked to EMPHASIZE: {user_emphasis}\n"
                "TYPE A — emphasize: this theme must appear in the summary AND in "
                "bullets of 2+ employers (3+ if there are 3+ employers), built "
                "ONLY from existing content — never invent supporting facts."
            ),
        ]

    if repair_instructions:
        lines += [
            "",
            "## repair_instructions (from the previous pass — you MUST fix these)",
            "- " + "\n- ".join(repair_instructions),
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------
def generate_aligned_content(
    resume_structure: ResumeStructure,
    jd_data: JobData,
    gap_analysis: GapAnalysis,
    *,
    original_resume_text: str,
    target_title: str | None = None,
    user_emphasis: str | None = None,
    repair_instructions: list[str] | None = None,
    on_token: TokenCallback | None = None,
) -> ResumeContent:
    """Generate the tailored resume. The Sonnet call is streamed (large output
    => timeout protection) and validated against `ResumeContent`."""
    user = build_generation_user(
        resume_structure,
        jd_data,
        gap_analysis,
        original_resume_text=original_resume_text,
        target_title=target_title,
        user_emphasis=user_emphasis,
        repair_instructions=repair_instructions,
    )
    # Combo gateways AUTO-think even when thinking isn't requested, and a full
    # resume rewrite burns a lot of that budget. The start budget is a measured
    # trade-off (see config.py): GATEWAY_START_TOKENS (64k) completes think+emit
    # in ONE attempt on the happy path, while a pathological auto-think is
    # capped early; starting at the 131k ceiling made the model think for 30+
    # MINUTES without emitting. If the 64k attempt still burns out on thinking
    # alone, tools/llm.py recovers exactly once at up to GATEWAY_MAX_TOKENS, and
    # every attempt is wall-clock-bounded there too.
    max_tokens = (
        config.GATEWAY_START_TOKENS if config.is_gateway() else config.DIRECT_MAX_TOKENS_GENERATE
    )
    return structured_call(
        model=config.MODEL_GENERATE,
        system=GENERATION_SYSTEM,
        user=user,
        output_model=ResumeContent,
        max_tokens=max_tokens,
        on_token=on_token,
        # Adaptive thinking is great on the direct API (real Sonnet 5). Combo
        # gateways auto-think regardless, so requesting it only adds risk of a
        # budget cutoff — skip it there.
        thinking=not config.is_gateway(),
    )


# ---------------------------------------------------------------------------
# Seniority depth floors (bug #6) — n8n PHASE 3 numbers
# ---------------------------------------------------------------------------
def seniority_bullet_floors(seniority: str, n_employers: int) -> list[int]:
    """Per-employer minimum bullets, most-recent first (n8n PHASE 3 numbers).

    senior: most recent >= 6, second >= 5, others >= 4
    mid:    most recent >= 4, second >= 4, others >= 3
    entry:  most recent >= 3, second >= 2, others >= 2
    """
    if seniority == "senior":
        base, tail = [6, 5], 4
    elif seniority == "entry":
        base, tail = [3, 2], 2
    else:  # mid
        base, tail = [4, 4], 3
    if n_employers <= len(base):
        return base[:n_employers]
    return base + [tail] * (n_employers - len(base))


def effective_bullet_count(emp) -> int:
    """Total work bullets for an employer: client bullets when has_clients."""
    if emp.has_clients and emp.clients:
        return sum(len(c.bullets or []) for c in emp.clients)
    return len(emp.bullets or [])


# ---------------------------------------------------------------------------
# Deterministic post-generation checks — code-enforced guarantees
# ---------------------------------------------------------------------------
def run_post_generation_checks(
    content: ResumeContent,
    jd_data: JobData,
    original_resume_text: str,
    seniority: str,
    *,
    resume_structure: ResumeStructure | None = None,
) -> tuple[ResumeContent, PostCheckReport]:
    """Run every deterministic guardrail on the generated content.

    Returns (possibly auto-fixed content, report). Grounding failures are
    auto-fixed (deterministic, safe); everything else becomes a repair
    instruction for the next pass.

    `resume_structure` (when given) supplies the parse-confirmed client names
    (bug #1's full contract): a client must be grounded AND have survived the
    parse-time >= MIN_BULLETS filter, otherwise it is flattened.
    """
    issues: list[Issue] = []

    # Bug #1 (defense in depth): flatten employers with ungrounded clients.
    valid_client_names = None
    if resume_structure is not None:
        valid_client_names = {
            c.name.strip().lower()
            for emp in resume_structure.employers
            for c in emp.clients
            if c.name.strip()
        }
    fixed = enforce_client_grounding(
        content.employers, original_resume_text, valid_client_names=valid_client_names
    )
    flattened = [
        e.name for e, f in zip(content.employers, fixed) if e.has_clients and not f.has_clients
    ]
    # Partial removal: employer still has_clients=True, but fewer clients
    # than before (a fabricated client was dropped without flattening
    # siblings that were still real — see enforce_client_grounding).
    partially_trimmed = [
        e.name
        for e, f in zip(content.employers, fixed)
        if e.has_clients and f.has_clients and len(f.clients) < len(e.clients)
    ]
    if flattened or partially_trimmed:
        content = content.model_copy(update={"employers": fixed})
        if flattened:
            issues.append(
                Issue(
                    code="client_grounding",
                    message=f"Flattened ungrounded clients under: {', '.join(flattened)}.",
                    repair="",  # already fixed; no regeneration needed
                )
            )
        if partially_trimmed:
            issues.append(
                Issue(
                    code="client_grounding",
                    message=f"Removed fabricated client(s) under: {', '.join(partially_trimmed)}.",
                    repair="",  # already fixed; no regeneration needed
                )
            )

    # Bug #4: keyword frequency cap, counted in code. Redundant skills/tools
    # tag mentions are trimmed automatically (safe — bullets keep the real
    # evidence); only keywords still over the cap after that need a repair
    # instruction, since removing THOSE requires rewriting prose.
    content, still_over = enforce_keyword_cap(content, jd_data)
    for kw, n in still_over:
        issues.append(
            Issue(
                code="keyword_cap",
                message=f"'{kw}' appears {n} times (max {config.MAX_KEYWORD_FREQ}).",
                repair=(
                    f"Reduce '{kw}' so it appears at most {config.MAX_KEYWORD_FREQ} "
                    f"times total across the resume (keep 1-2 in the summary/early "
                    f"bullets, drop the rest)."
                ),
            )
        )

    # Bug #3: contact completeness.
    missing = contact_completeness(content)
    if missing:
        issues.append(
            Issue(
                code="contact_completeness",
                message=f"Missing contact field(s): {', '.join(missing)}.",
                repair=(
                    f"Fill the missing contact field(s) ({', '.join(missing)}) with "
                    "the values from the candidate_resume_text contact header. "
                    "Never invent one if it is not in the source resume."
                ),
            )
        )

    # Bug #9: duplicate client attribution across employers.
    for name, owners in find_duplicate_clients(content):
        issues.append(
            Issue(
                code="duplicate_client",
                message=f"Client '{name}' attributed to multiple employers: {owners}.",
                repair=(
                    f"Keep '{name}' under only ONE employer (the one where the work "
                    f"is actually described); remove the copies from the others."
                ),
            )
        )

    # Bug #7 (auto-fix): collapse a comma-separated compound title down to
    # the primary title, before checking the word cap/uniqueness below.
    content, simplified_titles = enforce_single_role_title(content)
    if simplified_titles:
        issues.append(
            Issue(
                code="role_title_compound",
                message=f"Simplified compound role_title for: {', '.join(simplified_titles)}.",
                repair="",  # already fixed; no regeneration needed
            )
        )

    # Bug #7: role-title word cap + uniqueness.
    for emp_name, n in find_overlong_role_titles(content):
        issues.append(
            Issue(
                code="role_title_cap",
                message=f"'{emp_name}' role_title is {n} words (max {config.MAX_ROLE_TITLE_WORDS}).",
                repair=f"Shorten the role_title for '{emp_name}' to {config.MAX_ROLE_TITLE_WORDS} words or fewer.",
            )
        )
    for title in find_duplicate_role_titles(content):
        issues.append(
            Issue(
                code="role_title_unique",
                message=f"role_title '{title}' is reused by more than one employer.",
                repair=f"Reframe the role_title '{title}' so each employer has a unique title.",
            )
        )

    # Bug #6: seniority depth floors.
    floors = seniority_bullet_floors(seniority, len(content.employers))
    for i, emp in enumerate(content.employers):
        floor = floors[i] if i < len(floors) else floors[-1]
        n = effective_bullet_count(emp)
        if n < floor:
            issues.append(
                Issue(
                    code="seniority_depth",
                    message=f"'{emp.name}' has {n} bullets (floor {floor} for '{seniority}').",
                    repair=(
                        f"Expand '{emp.name}' to at least {floor} bullets (split/expand "
                        f"existing accomplishments; never invent metrics or facts)."
                    ),
                )
            )

    report = PostCheckReport(passed=not issues, issues=issues)
    return content, report


def summarize_differences(content: ResumeContent, jd_data: JobData) -> dict:
    """Deterministic before/after diff summary for the UI: which JD content
    the generated resume adds vs. reframes. Uses the gap analysis implicitly:
    anything missing before that now appears counts as 'added'."""
    text = collect_content_text(content).lower()
    added = [s for s in jd_data.required_skills if count_keyword_mentions(text, s) > 0]
    return {"added_skills": added, "total_required": len(jd_data.required_skills)}
