"""Tests for the deterministic post-generation checks.

These prove the hard guarantees that the n8n pipeline scattered across prompts
+ code nodes are now enforced in code after every generation.
"""
from __future__ import annotations

from schemas import (
    Certification,
    ContentClient,
    ContentContact,
    ContentEmployer,
    Education,
    EmployerProbe,
    JobData,
    ResumeContent,
    ResumeStructure,
    SkillCategory,
)
from tools.generate_content import (
    effective_bullet_count,
    run_post_generation_checks,
    seniority_bullet_floors,
)

JD = JobData(
    role_title="Senior Data Engineer",
    summary_of_role="Build pipelines.",
    required_skills=["Python", "SQL", "Spark", "Kubernetes"],
    key_responsibilities=["a", "b", "c", "d", "e", "f"],
    key_phrases=["data quality", "cloud-native"],
    outcomes=["lower costs"],
)

ORIGINAL = (
    "Jane Doe 555 Acme Corp 2020 Python SQL AWS Kafka pipeline reliability "
    "SameClient banking"
)


def _content(**kw) -> ResumeContent:
    base = ResumeContent(
        contact=ContentContact(name="Jane Doe", phone="555", email="j@c.com", linkedin="li/j"),
        summary="Engineer with 8 years in Python and SQL and data quality.",
        employers=[
            ContentEmployer(
                name="Acme Corp",
                start_date="2020",
                end_date="Present",
                role_title="Data Engineer",
                description="Built pipelines.",
                has_clients=False,
                bullets=[
                    "Designed Python and SQL pipelines.",
                    "Improved reliability with monitoring.",
                    "Cut costs with efficient storage.",
                    "Drove cloud-native adoption.",
                    "Championed data quality.",
                    "Partnered with analysts.",
                ],
            )
        ],
        skills=[SkillCategory(category="Core", items=["Python", "SQL", "Spark"])],
        certifications=[Certification(name="AWS Certified", year="2021")],
        education=[Education(institution="MIT", degree="BS", year="2015")],
    )
    base = base.model_copy(update=kw)
    return base


def test_clean_content_passes():
    content, report = run_post_generation_checks(_content(), JD, ORIGINAL, "senior")
    assert report.passed is True
    assert report.issues == []


def test_keyword_cap_flagged_when_over_limit():
    content = _content(summary="Python expert. Python architect. Python lead.")
    content, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    codes = [i.code for i in report.issues]
    assert "keyword_cap" in codes
    kw_issue = next(i for i in report.issues if i.code == "keyword_cap")
    assert "Python" in kw_issue.message
    assert "at most 3" in kw_issue.repair


def test_missing_contact_flagged():
    content = _content()
    content.contact = ContentContact(name="Jane Doe", email="j@c.com")  # no phone/linkedin
    content, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert any(i.code == "contact_completeness" for i in report.issues)


def test_duplicate_client_across_employers_flagged():
    content = _content()
    content.employers.append(
        ContentEmployer(
            name="Another Co",
            role_title="Engineer",
            has_clients=True,
            clients=[ContentClient(name="SameClient", bullets=["x", "y"])],
        )
    )
    # Put SameClient under both employers.
    content.employers[0].clients = [ContentClient(name="SameClient", bullets=["p", "q"])]
    content.employers[0].has_clients = True
    content.employers[0].bullets = []
    content, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert any(i.code == "duplicate_client" for i in report.issues)


def test_compound_role_title_auto_simplified():
    content = _content()
    content.employers[0].role_title = "Senior Data Engineer, Cloud ETL"
    fixed, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert fixed.employers[0].role_title == "Senior Data Engineer"
    assert any(i.code == "role_title_compound" for i in report.issues)
    # Auto-fixed, not left for the LLM to redo.
    compound_issue = next(i for i in report.issues if i.code == "role_title_compound")
    assert compound_issue.repair == ""


def test_hyphenated_role_title_untouched():
    content = _content()
    content.employers[0].role_title = "Full-Stack Engineer"
    fixed, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert fixed.employers[0].role_title == "Full-Stack Engineer"
    assert not any(i.code == "role_title_compound" for i in report.issues)


def test_role_title_overlong_flagged():
    content = _content()
    content.employers[0].role_title = "Senior Principal Lead Software Engineering Architect Extra"
    content, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert any(i.code == "role_title_cap" for i in report.issues)


def test_role_title_duplicate_flagged():
    content = _content()
    content.employers.append(
        ContentEmployer(name="Second Co", role_title="Data Engineer", has_clients=False, bullets=["a", "b", "c", "d", "e", "f"])
    )
    content, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert any(i.code == "role_title_unique" for i in report.issues)


def test_seniority_depth_flagged_for_thin_employer():
    thin = _content()
    thin.employers[0].bullets = ["only one bullet"]
    thin, report = run_post_generation_checks(thin, JD, ORIGINAL, "senior")
    assert any(i.code == "seniority_depth" for i in report.issues)
    assert "floor 6" in next(i.message for i in report.issues if i.code == "seniority_depth")


def test_grounding_auto_fixes_unreal_clients():
    content = _content()
    content.employers[0].has_clients = True
    content.employers[0].bullets = []
    content.employers[0].clients = [
        ContentClient(name="RealBank", bullets=["b1", "b2"]),
        ContentClient(name="MadeUp Corp", bullets=["f1", "f2"]),
    ]
    fixed, report = run_post_generation_checks(content, JD, ORIGINAL, "senior")
    assert fixed.employers[0].has_clients is False
    assert fixed.employers[0].clients == []
    assert any(i.code == "client_grounding" for i in report.issues)


def test_grounding_requires_parse_confirmed_clients():
    """Bug #1's full contract: grounding in the source text is not enough — a
    client must ALSO survive the parse-time >= MIN_BULLETS filter. A name that
    is merely mentioned once (the e2e 'Fabricated MegaCorp' trap) is grounded
    yet is not a real client, so it must be flattened when resume_structure
    confirms the parse did not keep it."""
    content = _content()
    content.employers[0].has_clients = True
    content.employers[0].bullets = []
    content.employers[0].clients = [
        # Grounded in ORIGINAL ("SameClient"), but the parse confirmed no clients.
        ContentClient(name="SameClient", bullets=["Delivered banking outcomes."]),
    ]
    structure = ResumeStructure(
        sections_present=["SUMMARY", "EXPERIENCE", "SKILLS"],
        employers=[
            EmployerProbe(name="Acme Corp", has_clients=False, clients=[], start_date="2020", end_date="Present")
        ],
        employer_count=1,
    )
    fixed, report = run_post_generation_checks(
        content, JD, ORIGINAL, "senior", resume_structure=structure
    )
    assert fixed.employers[0].has_clients is False
    assert fixed.employers[0].clients == []
    assert any(i.code == "client_grounding" for i in report.issues)


def test_seniority_bullet_floors():
    assert seniority_bullet_floors("senior", 3) == [6, 5, 4]
    assert seniority_bullet_floors("senior", 1) == [6]
    assert seniority_bullet_floors("mid", 2) == [4, 4]
    assert seniority_bullet_floors("entry", 4) == [3, 2, 2, 2]


def test_effective_bullet_count_uses_clients_when_has_clients():
    emp = ContentEmployer(
        name="Co",
        role_title="E",
        has_clients=True,
        clients=[ContentClient(name="A", bullets=["1", "2", "3"]), ContentClient(name="B", bullets=["4", "5"])],
    )
    assert effective_bullet_count(emp) == 5
    assert effective_bullet_count(_content().employers[0]) == 6
