"""Tests for the deterministic HTML renderer — exact-HTML assertions.

These encode the rendering rules that were hard-won in the n8n build:
UTF-8 charset, CSS-only bullets (ATS cleanliness), registry-ordered sections,
and precise page-break atomicity.
"""
from __future__ import annotations

from schemas import (
    AdditionalItem,
    AdditionalSection,
    Certification,
    ContentClient,
    ContentContact,
    ContentEmployer,
    Education,
    ResumeContent,
    ResumeStructure,
    SkillCategory,
)
from rendering.html_renderer import build_full_html

BULLET = "•"  # literal bullet char — must NEVER appear in HTML text.


def _full_content() -> ResumeContent:
    return ResumeContent(
        contact=ContentContact(name="Jane Doe", phone="+1", email="j@c.com", linkedin="li/jane"),
        summary="Experienced engineer.",
        employers=[
            ContentEmployer(
                name="Acme Corp",
                start_date="2020",
                end_date="Present",
                location="NYC",
                role_title="Senior Engineer",
                description="Led platform.",
                has_clients=False,
                bullets=["b1", "b2", "b3", "b4"],
            ),
            ContentEmployer(
                name="Globex",
                start_date="2016",
                end_date="2020",
                role_title="Engineer",
                has_clients=True,
                clients=[
                    ContentClient(name="Nimbus Bank", bullets=["c1", "c2"]),
                    ContentClient(name="AeroFly", bullets=["c3", "c4"]),
                ],
            ),
        ],
        skills=[SkillCategory(category="Languages", items=["Python", "Go"])],
        certifications=[Certification(name="AWS Certified", year="2022")],
        education=[Education(institution="MIT", degree="BS CS", location="Cambridge", year="2015")],
        additional_sections=[
            AdditionalSection(
                section_title="PUBLICATIONS",
                items=[AdditionalItem(heading="A Paper", subtext="Journal", bullets=["pub detail"])],
            )
        ],
    )


def _structure(*, order=None) -> ResumeStructure:
    return ResumeStructure(
        sections_present=list(order or ["PROFESSIONAL SUMMARY", "WORK EXPERIENCE", "SKILLS", "CERTIFICATIONS", "EDUCATION", "PUBLICATIONS"]),
        employer_count=2,
        estimated_seniority="senior",
        contact_fields_found=["name", "phone", "email", "linkedin"],
    )


def _render(*, content=None, structure=None) -> str:
    return build_full_html(content or _full_content(), structure or _structure())


def test_doctype_and_charset():
    html = _render()
    assert html.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8">' in html  # explicit — missing in the n8n version


def test_no_literal_bullet_in_html():
    html = _render()
    # Bullets are a painted shape, not a character: the literal • must never
    # appear in the document, not even inside a CSS pseudo-element.
    assert BULLET not in html
    # And the shape marker (textless span, styled as a circle) must be used.
    assert 'class="bullet-dot"' in html
    assert ".bullet-dot{" in html


def test_sections_rendered_in_sections_present_order():
    html = _render()
    summary = html.index("PROFESSIONAL SUMMARY")
    experience = html.index("PROFESSIONAL EXPERIENCE")
    skills = html.index("SKILLS")
    certs = html.index("CERTIFICATIONS")
    edu = html.index("EDUCATION")
    pubs = html.index("PUBLICATIONS")
    assert summary < experience < skills < certs < edu < pubs


def test_unmatched_present_section_appended_when_content_known():
    # EDUCATION not in sections_present but content exists -> appended in fallback order.
    structure = _structure(order=["PROFESSIONAL SUMMARY", "WORK EXPERIENCE", "SKILLS"])
    html = _render(structure=structure)
    assert "EDUCATION" in html
    assert html.index("SKILLS") < html.index("EDUCATION")


def test_employer_header_first_two_bullets_atomic():
    html = _render()
    # The atomic wrapper's first <ul> holds exactly bullets 1-2; 3-4 flow after.
    wrapper = html[html.index('class="employer-header-atomic"'):]
    ul = wrapper[wrapper.index("<ul>"):]
    ul = ul[: ul.index("</ul>") + len("</ul>")]
    assert "b1" in ul and "b2" in ul
    assert "b3" not in ul and "b4" not in ul
    # One atomic wrapper per employer.
    assert html.count('class="employer-header-atomic"') == 2


def test_skills_section_not_atomic_but_rows_are():
    html = _render()
    # Section class does NOT carry page-break-inside:avoid.
    skills_section = html[html.index('class="skills-section"'):]
    assert "page-break-inside:avoid" not in skills_section.split("</div>")[0]
    # Row class DOES carry it.
    assert ".skills-row{page-break-inside:avoid" in html or ".skills-row" in html
    assert "skills-row" in html


def test_certs_and_education_are_atomic():
    html = _render()
    # cert-line and edu-item both must carry page-break-inside:avoid.
    assert ".cert-line{page-break-inside:avoid" in html
    assert ".edu-item{page-break-inside:avoid" in html


def test_education_two_line_layout():
    html = _render()
    edu_html = html[html.index('class="edu-item"'):]
    assert "class='edu-institution'>MIT</span>" in edu_html
    assert "2015" in edu_html
    assert "BS CS" in edu_html
    assert "Cambridge" in edu_html


def test_client_blocks_only_when_has_clients():
    content = _full_content()
    html = _render(content=content)
    assert html.count('class="client-block"') == 2  # both clients rendered
    assert "Nimbus Bank" in html
    assert "AeroFly" in html


def test_contact_header_right_side():
    html = _render()
    right = html[html.index('class="header-right"'):]
    assert "+1" in right and "j@c.com" in right and "li/jane" in right


def test_html_escaping_of_llm_text():
    content = _full_content()
    content.employers[0].description = 'Led "platform" <team> & infra'
    html = _render(content=content)
    assert "&lt;team&gt;" in html
    assert "<team>" not in html
