"""Deterministic HTML renderer — port of the n8n 'Converting To HTML' code node.

No LLM here. Every rendering rule is hard-coded:
  - Explicit `charset="utf-8"` (missing in the n8n version — added deliberately).
  - Sections are rendered in `sections_present` ORDER via a normalized alias
    registry; unmatched additional sections are appended.
  - Employer header + first 2 bullets are atomic; remaining bullets flow.
  - Bullets are a painted CSS shape (`.bullet-dot`: an empty, textless
    `<span>` styled as a filled circle), never a text character — a
    `content:"•"` pseudo-element or a default `<ul>` disc marker both bake an
    actual "•" glyph into the rendered PDF's extractable text layer (verified
    live), which can trip up ATS parsers. A shape with no text node
    contributes nothing to that text layer while looking identical to the eye
    — a real round bullet, fully ATS-clean.
  - Skills section is NOT atomic (rows are); certs/education CAN be.
  - Education: institution+year on one line, degree+location on the next.
  - Body font stack lists "Liberation Sans" explicitly (not just a bare
    "sans-serif" generic) so the weasyprint/Pango PDF backend substitutes a
    real, metric-compatible Arial clone on environments without Arial itself
    (e.g. Streamlit Community Cloud) — Pango's generic-family fallback picks
    whatever default sans font a container happens to have, which produced a
    visibly worse, less professional PDF than the Playwright/Chromium path.
"""
from __future__ import annotations

import html
import re

from schemas import ResumeContent, ResumeStructure

# Section alias registry: normalized heading -> canonical rendering key.
# Rendering order always follows sections_present (the sole authority).
ALIASES: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "profile", "professional profile", "objective", "career objective", "about me"],
    "experience": ["experience", "professional experience", "work experience", "employment history", "work history", "professional background", "career history"],
    "skills": ["skills", "technical skills", "core skills", "skills summary", "skills & expertise"],
    "certifications": ["certifications", "certificates", "licenses", "professional certifications", "certification"],
    "education": ["education", "academic background", "educational background", "academics"],
}

DEFAULT_TITLES: dict[str, str] = {
    "summary": "PROFESSIONAL SUMMARY",
    "experience": "PROFESSIONAL EXPERIENCE",
    "skills": "SKILLS",
    "certifications": "CERTIFICATIONS",
    "education": "EDUCATION",
}

# Rendering order for canonical keys NOT seen in sections_present.
FALLBACK_ORDER: list[str] = ["summary", "experience", "skills", "certifications", "education"]

CSS = """
body{font-family:Arial,"Liberation Sans",sans-serif;font-size:9.5pt;color:#000;margin:0;padding:0;line-height:1.25}
.header-wrap{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:2px}
.header-name{font-size:20pt;font-weight:bold}
.header-title{font-size:9.5pt;font-style:italic;margin:2px 0 0 0}
.header-right{font-size:8.5pt;text-align:right;line-height:1.6}
hr{border:none;border-top:1.5px solid #000;margin:4px 0}
.section-wrap{page-break-inside:avoid}
.section-title{font-size:11pt;font-weight:bold;text-transform:uppercase;border-bottom:1px solid #000;margin:10px 0 5px 0;letter-spacing:0.5px;page-break-after:avoid}
.summary{margin:4px 0 0 0;line-height:1.35}
.employer{margin-bottom:8px}
.employer-header-atomic{page-break-inside:avoid}
.company-header{display:flex;justify-content:space-between;margin-top:6px}
.employer-name{font-weight:bold;font-size:10pt}
.employer-dates{font-weight:bold;font-size:8.5pt;white-space:nowrap}
.company-sub{display:flex;justify-content:space-between;align-items:baseline;margin-top:1px}
.role-title{font-style:italic;font-size:9pt}
.employer-loc{font-size:8.5pt}
.employer-desc{margin:2px 0 0 0;line-height:1.3}
ul{list-style:none;margin:3px 0 5px 0;padding-left:16px}
li{margin:0 0 2px 0;line-height:1.3;position:relative}
/* Painted shape, not a text character — visually a round bullet, but
   contributes nothing to the PDF's extractable text layer (ATS-clean).
   Absolutely positioned with an em-based offset (not inline-block +
   vertical-align): an inline-flow dot this small gets snapped to different
   fractional device pixels depending on surrounding text metrics, so
   Chromium anti-aliases each one to a visibly different size — this pins it
   to a fixed, font-relative position instead, so every dot renders identically. */
.bullet-dot{position:absolute;left:-14px;top:0.5em;width:4px;height:4px;border-radius:50%;background:#000}
.client-block{margin-left:12px;page-break-inside:avoid}
.client-name{font-weight:bold;font-style:italic;margin-top:3px;position:relative;padding-left:12px}
.client-name .bullet-dot{left:0}
.client-desc{margin:1px 0 0 0;line-height:1.3;font-style:italic}
.skills-section{margin-bottom:6px}
.skills-row{page-break-inside:avoid;margin:2px 0}
.skill-category{font-weight:bold}
.cert-line{page-break-inside:avoid;display:flex;justify-content:space-between;margin:2px 0}
.cert-name{font-weight:bold}
.edu-item{page-break-inside:avoid;margin:3px 0}
.edu-header{display:flex;justify-content:space-between;align-items:baseline}
.edu-institution{font-weight:bold}
.edu-year{font-style:italic;white-space:nowrap}
.edu-degree{font-style:italic;margin-left:14px}
.additional-item{page-break-inside:avoid;margin:3px 0}
.additional-heading{font-weight:bold}
.additional-subtext{font-style:italic;margin:1px 0 0 14px}
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _alias_key(heading: str) -> str | None:
    h = _norm(heading)
    for key, aliases in ALIASES.items():
        if any(_norm(a) == h for a in aliases):
            return key
    return None


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _bullets(items: list[str]) -> str:
    if not items:
        return ""
    # The marker is a painted CSS shape (empty span, no text node), never a
    # character — a `content:"•"` (or default <ul> disc) pseudo-element bakes
    # an actual "•" glyph into the PDF's extractable text layer (verified
    # live), which an ATS parser can choke on. A borderless, textless
    # <span> painted as a circle renders identically to the eye but
    # contributes zero characters to that text layer.
    lis = "".join(f'<li><span class="bullet-dot"></span>{_esc(b)}</li>' for b in items)
    return f"<ul>{lis}</ul>"


def _render_contact(content: ResumeContent, title: str = "") -> str:
    c = content.contact
    right = [x for x in [c.phone, c.email, c.linkedin] if x.strip()]
    right_html = "<br>".join(_esc(r) for r in right)
    title_html = f'<div class="header-title">{_esc(title)}</div>' if title.strip() else ""
    return (
        f'<div class="header-wrap">'
        f'<div><div class="header-name">{_esc(c.name)}</div>{title_html}</div>'
        f'<div class="header-right">{right_html}</div>'
        f"</div><hr>"
    )


def _render_summary(content: ResumeContent) -> str:
    if not content.summary.strip():
        return ""
    return (
        '<div class="section-wrap">'
        f'<div class="section-title">{_esc(DEFAULT_TITLES["summary"])}</div>'
        f'<p class="summary">{_esc(content.summary)}</p>'
        "</div>"
    )


def _render_employers(content: ResumeContent) -> str:
    if not content.employers:
        return ""
    blocks = []
    for emp in content.employers:
        # Header + description + first 2 employer bullets are atomic.
        # Row 1: company name (left) + dates (right). Row 2: role title
        # (italic, left) + location (right).
        header = (
            '<div class="company-header">'
            f'<span class="employer-name">{_esc(emp.name)}</span>'
            f'<span class="employer-dates">{_esc(emp.start_date)}&thinsp;–&thinsp;{_esc(emp.end_date)}</span>'
            "</div>"
        )
        sub = (
            '<div class="company-sub">'
            f'<span class="role-title">{_esc(emp.role_title)}</span>'
            f'<span class="employer-loc">{_esc(emp.location)}</span>'
            "</div>"
            if emp.role_title.strip() or emp.location.strip()
            else ""
        )
        desc = f'<p class="employer-desc">{_esc(emp.description)}</p>' if emp.description.strip() else ""
        bullets = list(emp.bullets or [])
        atomic = f'<div class="employer-header-atomic">{header}{sub}{desc}{_bullets(bullets[:2])}</div>'
        rest = _bullets(bullets[2:])

        clients = ""
        if emp.has_clients and emp.clients:
            for cl in emp.clients:
                clients += (
                    '<div class="client-block">'
                    f'<div class="client-name"><span class="bullet-dot"></span>{_esc(cl.name)}</div>'
                    + (f'<div class="client-desc">{_esc(cl.description)}</div>' if cl.description.strip() else "")
                    + _bullets(list(cl.bullets or []))
                    + "</div>"
                )

        blocks.append(f'<div class="employer">{atomic}{rest}{clients}</div>')
    title = f'<div class="section-title">{_esc(DEFAULT_TITLES["experience"])}</div>'
    return title + "".join(blocks)


def _render_skills(content: ResumeContent) -> str:
    if not content.skills:
        return ""
    rows = []
    for cat in content.skills:
        items = ", ".join(_esc(i) for i in cat.items)
        if not items:
            continue
        label = f"<span class='skill-category'>{_esc(cat.category)}:</span> " if cat.category.strip() else ""
        rows.append(f'<div class="skills-row">{label}<span>{items}</span></div>')
    if not rows:
        return ""
    # Section deliberately NOT atomic — flows across pages; rows are atomic.
    return (
        '<div class="skills-section">'
        f'<div class="section-title">{_esc(DEFAULT_TITLES["skills"])}</div>'
        + "".join(rows)
        + "</div>"
    )


def _render_certifications(content: ResumeContent) -> str:
    if not content.certifications:
        return ""
    lines = []
    for cert in content.certifications:
        year = f"<span>{_esc(cert.year)}</span>" if cert.year.strip() else ""
        lines.append(f'<div class="cert-line"><span class="cert-name">{_esc(cert.name)}</span>{year}</div>')
    return (
        '<div class="section-wrap">'
        f'<div class="section-title">{_esc(DEFAULT_TITLES["certifications"])}</div>'
        + "".join(lines)
        + "</div>"
    )


def _render_education(content: ResumeContent) -> str:
    if not content.education:
        return ""
    items = []
    for edu in content.education:
        year = f"<span class='edu-year'>{_esc(edu.year)}</span>" if edu.year.strip() else ""
        degree = f"<span class='edu-degree'>{_esc(edu.degree)}</span>" if edu.degree.strip() else ""
        loc = f"<span> — {_esc(edu.location)}</span>" if edu.location.strip() else ""
        items.append(
            '<div class="edu-item">'
            '<div class="edu-header">'
            f"<span class='edu-institution'>{_esc(edu.institution)}</span>{year}"
            "</div>"
            f"{degree}{loc}"
            "</div>"
        )
    return (
        '<div class="section-wrap">'
        f'<div class="section-title">{_esc(DEFAULT_TITLES["education"])}</div>'
        + "".join(items)
        + "</div>"
    )


def _render_additional(sec_title: str, content: ResumeContent) -> str:
    """Render an additional section whose normalized title matches `sec_title`
    (or render the first unmatched additional section for a present heading)."""
    target = _norm(sec_title)
    match = next(
        (s for s in content.additional_sections if _norm(s.section_title) == target),
        None,
    )
    if match is None:
        return ""
    items = []
    for it in match.items:
        heading = f'<div class="additional-heading">{_esc(it.heading)}</div>' if it.heading.strip() else ""
        subtext = f'<div class="additional-subtext">{_esc(it.subtext)}</div>' if it.subtext.strip() else ""
        items.append(
            f'<div class="additional-item">{heading}{subtext}{_bullets(list(it.bullets or []))}</div>'
        )
    return (
        '<div class="section-wrap">'
        f'<div class="section-title">{_esc(match.section_title)}</div>'
        + "".join(items)
        + "</div>"
    )


def render_resume_html(content: ResumeContent, structure: ResumeStructure, title: str = "") -> str:
    """Render the full resume document (sans `<head>` — caller wraps it).

    `title` is the target job title shown as a subtitle under the candidate's
    name (n8n's header-title, sourced from the form's "Job Role" field)."""
    # 1. Ordered list of (kind, heading) — driven by sections_present.
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    matched_additional: set[str] = set()

    for heading in structure.sections_present:
        key = _alias_key(heading)
        if key is not None:
            if key not in seen:
                ordered.append((key, heading))
                seen.add(key)
        else:
            # Maybe an additional section with this exact title exists.
            target = _norm(heading)
            hit = next(
                (s.section_title for s in content.additional_sections if _norm(s.section_title) == target),
                None,
            )
            if hit is not None:
                ordered.append(("additional", hit))
                matched_additional.add(_norm(hit))
            # else: no content — skip silently (validation guards completeness).

    # 2. Registry keys present in content but not in sections_present → append.
    has_content = {
        "summary": bool(content.summary.strip()),
        "experience": bool(content.employers),
        "skills": bool(content.skills),
        "certifications": bool(content.certifications),
        "education": bool(content.education),
    }
    for key in FALLBACK_ORDER:
        if key not in seen and has_content.get(key):
            ordered.append((key, DEFAULT_TITLES[key]))

    # 3. Additional sections the generator emitted but no present heading named → append.
    for sec in content.additional_sections:
        if _norm(sec.section_title) not in matched_additional:
            ordered.append(("additional", sec.section_title))

    # 4. Render in order.
    parts: list[str] = [_render_contact(content, title)]
    for kind, heading in ordered:
        if kind == "summary":
            parts.append(_render_summary(content))
        elif kind == "experience":
            parts.append(_render_employers(content))
        elif kind == "skills":
            parts.append(_render_skills(content))
        elif kind == "certifications":
            parts.append(_render_certifications(content))
        elif kind == "education":
            parts.append(_render_education(content))
        elif kind == "additional":
            parts.append(_render_additional(heading, content))

    return "\n".join(p for p in parts if p)


def build_full_html(content: ResumeContent, structure: ResumeStructure, title: str = "") -> str:
    """Return the complete, standalone HTML document (with `<head>`)."""
    body = render_resume_html(content, structure, title)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Resume</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n"
        f"<body>\n{body}\n</body>\n"
        "</html>"
    )
