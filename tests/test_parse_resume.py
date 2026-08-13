"""Tests for `parse_resume_structure` — LLM extraction + deterministic filter.

The LLM call is mocked; these tests verify the plumbing: correct prompt is
built, the structured call is made with the right model/schema, and — critically —
the returned structure is ALWAYS post-processed by the client-detection filter,
even when the LLM hallucinates clients.
"""
from __future__ import annotations

from unittest.mock import patch

from schemas import ClientProbe, EmployerProbe, ResumeStructure
from tools.parse_resume import parse_resume_structure

SAMPLE_RESUME = "Jane Doe\nProfessional Summary: ...\nWork Experience:\nAcme Corp 2020-present\n..."

HALLUCINATED_STRUCTURE = ResumeStructure(
    sections_present=["PROFESSIONAL SUMMARY", "WORK EXPERIENCE"],
    employers=[
        EmployerProbe(
            name="Acme Corp",
            has_clients=True,
            clients=[
                ClientProbe(name="Real Bank", bullet_count=3),
                ClientProbe(name="Fake Client Inc", bullet_count=1),  # must be filtered
            ],
        )
    ],
    employer_count=1,
    estimated_seniority="mid",
    contact_fields_found=["name", "email"],
)


def test_parse_applies_client_filter_to_llm_output():
    with patch("tools.parse_resume.structured_call", return_value=HALLUCINATED_STRUCTURE) as call:
        result = parse_resume_structure(SAMPLE_RESUME)

    # Deterministic guardrail ran on top of the (mocked) LLM proposal.
    emp = result.employers[0]
    assert [c.name for c in emp.clients] == ["Real Bank"]
    assert emp.has_clients is True
    assert result.employer_count == 1

    # Correct model + schema were used.
    _, kwargs = call.call_args
    assert kwargs["output_model"] is ResumeStructure
    assert kwargs["max_tokens"] == 4000


def test_parse_calls_with_resume_text_in_user_message():
    with patch("tools.parse_resume.structured_call") as call:
        parse_resume_structure("My custom resume text")

    _, kwargs = call.call_args
    user = kwargs["user"]
    assert "My custom resume text" in user


def test_parse_flattens_employer_without_real_clients():
    fake = ResumeStructure(
        sections_present=["WORK EXPERIENCE"],
        employers=[
            EmployerProbe(
                name="Acme Corp",
                has_clients=True,
                clients=[
                    ClientProbe(name="Mentioned In Passing", bullet_count=0),
                    ClientProbe(name="Another Passing", bullet_count=0),
                ],
            )
        ],
        employer_count=1,
        estimated_seniority="entry",
        contact_fields_found=["name"],
    )
    with patch("tools.parse_resume.structured_call", return_value=fake):
        result = parse_resume_structure(SAMPLE_RESUME)

    assert result.employers[0].has_clients is False
    assert result.employers[0].clients == []
