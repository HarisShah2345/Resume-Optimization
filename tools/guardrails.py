"""Deterministic safety checks ported from the n8n code nodes.

The LLM can PROPOSE client structures, keyword usage, employer sets — but
these functions VERIFY and ENFORCE. Each check exists because a real bug
produced a bad resume. No LLM call happens here; everything is pure and
unit-testable without an API key.
"""
from __future__ import annotations

import re

import config
from schemas import ContentEmployer, EmployerProbe, JobData, ResumeContent


# ---------------------------------------------------------------------------
# Bug #1 — fabricated client sub-structures
# Port of the 'Fetching Output1' n8n code node.
# ---------------------------------------------------------------------------
def filter_clients_by_min_bullets(
    employers: list[EmployerProbe],
    min_bullets: int | None = None,
) -> list[EmployerProbe]:
    """Drop clients with fewer than `min_bullets` individually-attributed
    sentences. If no client survives for an employer, flatten it:
    `has_clients=False, clients=[]`.

    The LLM's `bullet_count` already excludes collective mentions (0 per
    client), so this filter is the hard guarantee that a client block only
    appears when there is real, attributed evidence for it.
    """
    min_bullets = min_bullets if min_bullets is not None else config.MIN_BULLETS_PER_CLIENT
    cleaned: list[EmployerProbe] = []
    for emp in employers:
        real_clients = [c for c in emp.clients if c.bullet_count >= min_bullets]
        cleaned.append(
            emp.model_copy(update={"has_clients": bool(real_clients), "clients": real_clients})
        )
    return cleaned


# ---------------------------------------------------------------------------
# Bug #1 (defense-in-depth) — ground-truth grounding of client names.
# Port of 'isGroundedClientName' + 'Fetching output 2' n8n code nodes.
# ---------------------------------------------------------------------------
def is_grounded(name: str, resume_text: str, min_len: int = 2) -> bool:
    """True if `name` (cleaned: lowercased, trimmed, len >= min_len) appears in
    `resume_text` as a whole word. Word boundaries prevent matching inside an
    unrelated word (e.g. "ai" matching "chair").

    Whitespace between the name's words is matched as `\\s+`, not a literal
    space — `resume_text` usually comes from `pypdf.extract_text()` on an
    uploaded PDF, which routinely turns a single space between two words into
    a newline (or several spaces) when they wrapped across a visual line or a
    column boundary. A multi-word client name is exactly the kind of text
    that wraps, so an exact-space match was flattening EVERY employer's real
    clients on real uploaded resumes (verified live) — a false positive on
    the fabrication guardrail, not a real fabrication."""
    clean = re.sub(r"\s+", " ", name.strip().lower())
    if len(clean) < min_len:
        return False
    words = [re.escape(w) for w in clean.split(" ")]
    pattern = re.compile(r"(?<![a-z0-9])" + r"\s+".join(words) + r"(?![a-z0-9])")
    return bool(pattern.search(resume_text.lower()))


def enforce_client_grounding(
    employers: list[ContentEmployer],
    resume_text: str,
    valid_client_names: set[str] | None = None,
) -> list[ContentEmployer]:
    """Remove each client that is NOT grounded (or not parse-confirmed) from
    its employer INDIVIDUALLY — a fabricated client sitting next to a real
    one must never take the real one down with it.

    This used to flatten the WHOLE employer (merge every client's bullets up,
    `has_clients=False`) the moment ANY one client failed. That was too
    aggressive: verified live, a Haiku validation pass once hallucinated a
    replacement name for a real client ("Associated Bank" -> "AWS Glue
    Migration Project"), and the old behavior wiped out the untouched,
    still-real "Bank of America" client sitting right next to it in the same
    employer. Only the fabricated client's own bullets/tools are discarded —
    they are NOT merged into employer-level `bullets`/`tools` while
    `has_clients` stays True, since the renderer only reads those fields when
    `has_clients` is False; merging would just make that content invisible.

    If NO client survives, the employer is still flattened to
    `has_clients=False` exactly as before, merging every client's bullets/
    tools up so the underlying accomplishments are never lost when there's no
    other structure left to hold them.

    This is the last line of defense against fabricated client blocks before
    validation. `resume_text` is the ORIGINAL resume (ground truth), never the
    generated text.

    `valid_client_names` (lowercased) is the set of clients the PARSE stage
    confirmed with >= `MIN_BULLETS_PER_CLIENT` individually-attributed sentences
    (bug #1's full contract). Grounding in the raw text is necessary but not
    sufficient: a name that appears only as a single passing mention (the e2e
    "Fabricated MegaCorp" trap) IS grounded, yet is not a real client — without
    the valid-name check it would survive as a promoted client block.
    """

    def _ok(c) -> bool:
        return is_grounded(c.name, resume_text) and (
            valid_client_names is None or c.name.strip().lower() in valid_client_names
        )

    cleaned: list[ContentEmployer] = []
    for emp in employers:
        if emp.has_clients and emp.clients:
            real = [c for c in emp.clients if _ok(c)]
            if len(real) == len(emp.clients):
                cleaned.append(emp)
            elif real:
                cleaned.append(emp.model_copy(update={"clients": real}))
            else:
                merged_bullets = list(emp.bullets or [])
                merged_tools = list(emp.tools or [])
                for c in emp.clients:
                    merged_bullets.extend(c.bullets or [])
                    merged_tools.extend(c.tools or [])
                cleaned.append(
                    emp.model_copy(
                        update={
                            "has_clients": False,
                            "clients": [],
                            "bullets": merged_bullets,
                            "tools": merged_tools,
                        }
                    )
                )
        else:
            cleaned.append(emp)
    return cleaned


# ---------------------------------------------------------------------------
# Bug #4 — keyword frequency discipline. Counted in code, never LLM self-report.
# ---------------------------------------------------------------------------
def count_keyword_mentions(text: str, keyword: str) -> int:
    """Case-insensitive whole-word count of `keyword` in `text` (may be a
    multi-word phrase; boundaries apply around the whole phrase)."""
    k = keyword.strip().lower()
    if not k:
        return 0
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])")
    return len(pattern.findall(text.lower()))


def collect_content_text(content: ResumeContent) -> str:
    """Aggregate every piece of generated text (summary, employers, clients,
    skills, certs, education, additional sections) into one searchable string
    for frequency counting."""
    parts: list[str] = [content.summary]
    for emp in content.employers:
        parts.extend([emp.name, emp.role_title, emp.description])
        parts.extend(emp.bullets or [])
        parts.extend(emp.tools or [])
        for cl in emp.clients:
            parts.extend([cl.name, cl.description])
            parts.extend(cl.bullets or [])
            parts.extend(cl.tools or [])
    for cat in content.skills:
        parts.append(cat.category)
        parts.extend(cat.items or [])
    for cert in content.certifications:
        parts.extend([cert.name, cert.year])
    for edu in content.education:
        parts.extend([edu.institution, edu.degree, edu.location, edu.year])
    for sec in content.additional_sections:
        parts.append(sec.section_title)
        for item in sec.items:
            parts.extend([item.heading, item.subtext])
            parts.extend(item.bullets or [])
    return "\n".join(p for p in parts if p)


def find_over_limit_keywords(
    content: ResumeContent,
    jd_data: JobData,
    max_freq: int | None = None,
) -> list[tuple[str, int]]:
    """Return [(keyword, actual_count)] for every JD keyword used more than
    `max_freq` times across the whole generated resume. Empty list == passing."""
    max_freq = max_freq if max_freq is not None else config.MAX_KEYWORD_FREQ
    text = collect_content_text(content)
    over: list[tuple[str, int]] = []
    for kw in jd_data.required_skills + jd_data.key_phrases:
        if not kw:
            continue
        n = count_keyword_mentions(text, kw)
        if n > max_freq:
            over.append((kw, n))
    return over


def enforce_keyword_cap(
    content: ResumeContent,
    jd_data: JobData,
    max_freq: int | None = None,
) -> tuple[ResumeContent, list[tuple[str, int]]]:
    """Deterministically trim over-limit JD keywords by removing redundant
    TAG-style mentions — skills-category items and employer/client tools
    lists — never bullet prose, so no sentence is ever mangled.

    Bullets carry the actual evidence for a skill; the same term commonly ALSO
    appears as a standalone tag in skills/tools. Once bullets alone already
    hit the cap, the tag is pure keyword-stuffing repetition (bug #4's whole
    point) and dropping it removes no claim. Verified live: this is exactly
    the failure mode the LLM repair pass couldn't reliably converge on
    (asking the model to "reduce X to <=3 mentions" burned 100-300s of
    thinking and still missed 3/3 times) — trimming a tag list is unambiguous
    and needs no LLM judgment at all.

    Returns (possibly-fixed content, [(keyword, count)] still over the cap
    after trimming — these still need a repair instruction, since they live
    only in bullet prose that a human/LLM edit must actually rewrite).
    Returns `content` unchanged (same identity) when nothing is over the cap.
    """
    max_freq = max_freq if max_freq is not None else config.MAX_KEYWORD_FREQ
    if not find_over_limit_keywords(content, jd_data, max_freq):
        return content, []
    skills = [cat.model_copy(update={"items": list(cat.items or [])}) for cat in content.skills]
    employers = [
        emp.model_copy(
            update={
                "tools": list(emp.tools or []),
                "clients": [
                    cl.model_copy(update={"tools": list(cl.tools or [])}) for cl in emp.clients
                ],
            }
        )
        for emp in content.employers
    ]

    def current_text() -> str:
        return collect_content_text(
            content.model_copy(update={"skills": skills, "employers": employers})
        )

    def trim(kw: str, deficit: int) -> None:
        tag_lists = [cat.items for cat in skills]
        tag_lists += [emp.tools for emp in employers]
        tag_lists += [cl.tools for emp in employers for cl in emp.clients]
        for items in tag_lists:
            i = 0
            while i < len(items) and deficit > 0:
                n = count_keyword_mentions(items[i], kw)
                if n > 0:
                    deficit -= n
                    del items[i]
                    continue
                i += 1
            if deficit <= 0:
                return

    still_over: list[tuple[str, int]] = []
    for kw in jd_data.required_skills + jd_data.key_phrases:
        if not kw:
            continue
        n = count_keyword_mentions(current_text(), kw)
        if n <= max_freq:
            continue
        trim(kw, n - max_freq)
        n = count_keyword_mentions(current_text(), kw)
        if n > max_freq:
            still_over.append((kw, n))

    fixed = content.model_copy(update={"skills": skills, "employers": employers})
    return fixed, still_over


# ---------------------------------------------------------------------------
# Bug #9 — duplicate client attribution across employers.
# ---------------------------------------------------------------------------
def find_duplicate_clients(content: ResumeContent) -> list[tuple[str, list[str]]]:
    """Return [(client_name, [employer_names...])] for any client attributed
    under more than one employer. The wrong-employer copies must be removed."""
    owner: dict[str, list[str]] = {}
    for emp in content.employers:
        for cl in emp.clients or []:
            key = cl.name.strip().lower()
            if not key:
                continue
            owner.setdefault(key, []).append(emp.name)
    return [(name, owners) for name, owners in owner.items() if len(owners) > 1]


# ---------------------------------------------------------------------------
# Bug #3 — contact completeness (deterministic presence check).
# ---------------------------------------------------------------------------
def contact_completeness(content: ResumeContent) -> list[str]:
    """Return the list of missing contact fields among
    [name, phone, email, linkedin]. Empty list == complete."""
    c = content.contact
    missing = []
    if not c.name.strip():
        missing.append("name")
    if not c.phone.strip():
        missing.append("phone")
    if not c.email.strip():
        missing.append("email")
    if not c.linkedin.strip():
        missing.append("linkedin")
    return missing


# ---------------------------------------------------------------------------
# Bug #7 — role-title reframe cap. Never overclaim; never exceed the word cap.
# ---------------------------------------------------------------------------
def enforce_single_role_title(content: ResumeContent) -> tuple[ResumeContent, list[str]]:
    """Deterministically collapse a compound, comma-separated role title
    ('Senior Data Engineer, Cloud ETL') down to the primary title before the
    comma ('Senior Data Engineer').

    The word cap alone doesn't catch this — a comma-joined compound is often
    still <= MAX_ROLE_TITLE_WORDS, so it silently passed `find_overlong_role_titles`
    while still looking like two roles bolted together (verified live: this
    survived generation repeatedly regardless of the prompt asking for a
    single title). Only splits on a comma — never a hyphen — so a genuinely
    hyphenated title ('Full-Stack Engineer') is never touched.

    Returns (possibly-fixed content, [employer names] that were simplified).
    """
    changed: list[str] = []
    fixed_employers = []
    for emp in content.employers:
        title = emp.role_title
        if "," in title:
            simplified = title.split(",", 1)[0].strip()
            if simplified:
                changed.append(emp.name)
                fixed_employers.append(emp.model_copy(update={"role_title": simplified}))
                continue
        fixed_employers.append(emp)
    if not changed:
        return content, []
    return content.model_copy(update={"employers": fixed_employers}), changed


def find_overlong_role_titles(
    content: ResumeContent,
    max_words: int | None = None,
) -> list[tuple[str, int]]:
    """Return [(employer_name, word_count)] for role titles exceeding the cap.
    Empty list == passing."""
    max_words = max_words if max_words is not None else config.MAX_ROLE_TITLE_WORDS
    overlong = []
    for emp in content.employers:
        n = len(emp.role_title.split())
        if n > max_words:
            overlong.append((emp.name, n))
    return overlong


def find_duplicate_role_titles(content: ResumeContent) -> list[str]:
    """Return role titles reused verbatim by more than one employer (each role
    must be reframed uniquely — bug #7)."""
    seen: dict[str, int] = {}
    for emp in content.employers:
        title = emp.role_title.strip().lower()
        if title:
            seen[title] = seen.get(title, 0) + 1
    return [t for t, n in seen.items() if n > 1]
