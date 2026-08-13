"""Shared Pydantic v2 models — the contracts every tool, the agent state, and
the renderer agree on. Mirrors the JSON schemas from the original n8n pipeline.

These models are used for BOTH structured-output validation (via the
`anthropic` SDK's JSON-schema tool) and deterministic post-processing.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Parse stage
# ---------------------------------------------------------------------------


class ClientProbe(BaseModel):
    """A named client detected by the LLM during structure parsing.

    `bullet_count` counts ONLY sentences individually attributed to this client.
    Collective mentions ("partners including X, Y and Z") are 0 for each client.
    """

    name: str = Field(description="Exact client/company name as it appears in the resume.")
    bullet_count: int = Field(
        description=(
            "Number of resume sentences individually attributed to this client. "
            "Collective mentions count as 0."
        )
    )


class EmployerProbe(BaseModel):
    name: str = Field(description="Employer name exactly as it appears in the resume.")
    has_clients: bool = Field(
        description=(
            "True ONLY if >= 2 distinct named clients each have >= 2 individually "
            "attributed achievements. Otherwise false."
        )
    )
    clients: list[ClientProbe] = Field(
        default_factory=list,
        description="Named clients detected under this employer (generic or passing mentions excluded).",
    )
    start_date: str | None = None
    end_date: str | None = None


class ResumeStructure(BaseModel):
    """Structured view of the original resume (LLM-extracted, code-sanitized).

    `sections_present` is the SOLE AUTHORITY for section order and employer set
    (bug #2): generated output must match it 1:1.
    """

    sections_present: list[str] = Field(
        description="Section headings in order of appearance, e.g. ['PROFESSIONAL SUMMARY','WORK EXPERIENCE','SKILLS']."
    )
    employers: list[EmployerProbe] = Field(
        default_factory=list,
        description="One entry per employer, in order of appearance.",
    )
    employer_count: int = Field(default=0)
    has_projects_section: bool = False
    has_publications_section: bool = False
    estimated_seniority: Literal["entry", "mid", "senior"] = "mid"
    contact_fields_found: list[str] = Field(
        default_factory=list,
        description="Which of [name, phone, email, linkedin] were found.",
    )


# ---------------------------------------------------------------------------
# Job description parse
# ---------------------------------------------------------------------------


class JobData(BaseModel):
    role_title: str = Field(description="Exact job title from the posting.")
    summary_of_role: str = Field(description="1-2 sentence plain-language summary of what the role does.")
    required_skills: list[str] = Field(
        default_factory=list,
        description="Required skills (>= 6 items).",
    )
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="Key responsibilities (>= 6 items).",
    )
    preferred_qualifications: list[str] = Field(default_factory=list)
    key_phrases: list[str] = Field(
        default_factory=list,
        description="Verbatim phrases/quotes from the posting (>= 5 items).",
    )
    stakeholders: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(
        default_factory=list,
        description="Business outcomes the role is expected to drive (>= 3 items).",
    )


# ---------------------------------------------------------------------------
# Alignment gap (deterministic)
# ---------------------------------------------------------------------------

GAP_SEVERITY = Literal["full_rebuild", "heavy_tailoring", "light_tweak"]


class GapAnalysis(BaseModel):
    severity: GAP_SEVERITY = Field(
        description=(
            "full_rebuild: resume lacks most JD content; heavy_tailoring: moderate "
            "gaps to add/reframe; light_tweak: strong fit, minor polish."
        )
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Required skills absent from the resume keywords.",
    )
    missing_phrases: list[str] = Field(
        default_factory=list,
        description="Key JD phrases absent from the resume text.",
    )
    missing_outcomes: list[str] = Field(
        default_factory=list,
        description="Outcomes with no supporting evidence in the resume.",
    )
    depth_target: Literal["entry", "mid", "senior"] = "mid"
    reasoning: str = Field(
        default="",
        description="Human-readable one-paragraph reasoning for the agent + UI.",
    )


# ---------------------------------------------------------------------------
# Generation / validation output (also consumed by the renderer)
# ---------------------------------------------------------------------------


class ContentContact(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""
    linkedin: str = ""


class ContentClient(BaseModel):
    name: str
    description: str = ""
    bullets: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class ContentEmployer(BaseModel):
    name: str
    start_date: str = ""
    end_date: str = ""
    location: str = ""
    role_title: str = Field(
        description="Reframed role title, <= 6 words, unique per employer, truthful (bug #7)."
    )
    description: str = ""
    has_clients: bool = False
    bullets: list[str] = Field(
        default_factory=list,
        description="Used only when has_clients is False.",
    )
    tools: list[str] = Field(default_factory=list)
    clients: list[ContentClient] = Field(default_factory=list)


class SkillCategory(BaseModel):
    category: str = ""
    items: list[str] = Field(default_factory=list)


class Certification(BaseModel):
    name: str = ""
    year: str = ""


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    location: str = ""
    year: str = ""


class AdditionalItem(BaseModel):
    heading: str = ""
    subtext: str = ""
    bullets: list[str] = Field(default_factory=list)


class AdditionalSection(BaseModel):
    section_title: str = ""
    items: list[AdditionalItem] = Field(default_factory=list)


class ResumeContent(BaseModel):
    contact: ContentContact = Field(default_factory=ContentContact)
    summary: str = Field(
        default="",
        description="Summary paragraph using exact JD key_phrases (bug #5).",
    )
    employers: list[ContentEmployer] = Field(default_factory=list)
    skills: list[SkillCategory] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    additional_sections: list[AdditionalSection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Post-generation guardrail reports (deterministic, code-enforced)
# ---------------------------------------------------------------------------


class Issue(BaseModel):
    """A single deterministic violation found after generation."""

    code: str = Field(description="Short stable code, e.g. 'keyword_cap', 'client_grounding'.")
    message: str
    # A repair instruction the agent can feed back into the next generation pass.
    repair: str = ""


class PostCheckReport(BaseModel):
    passed: bool = False
    issues: list[Issue] = Field(default_factory=list)

    @property
    def repair_instructions(self) -> list[str]:
        return [i.repair for i in self.issues if i.repair]
