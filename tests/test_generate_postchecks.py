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
    GapAnalysis,
    JobData,
    ResumeContent,
    ResumeStructure,
    SkillCategory,
)
import config
from tools.generate_content import (
    build_generation_user,
    effective_bullet_count,
    generate_aligned_content,
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


# ---------------------------------------------------------------------------
# build_generation_user — prompt construction (lines 113-161)
# ---------------------------------------------------------------------------
from schemas import ResumeStructure  # noqa: E402

_PROBE_STRUCTURE = ResumeStructure(
    sections_present=["SUMMARY", "EXPERIENCE", "SKILLS", "CERTIFICATIONS", "EDUCATION"],
    employers=[],
    employer_count=1,
    estimated_seniority="senior",
    total_bullets=6,
)

_PROBE_GAP = GapAnalysis(
    severity="light_tweak",
    reasoning="Test",
    missing_skills=["Kubernetes"],
    missing_phrases=["cloud-native"],
    missing_responsibilities=["orchestrate"],
    missing_outcomes=["reduce costs"],
    depth_target="senior",
)


def test_build_generation_user_basic():
    """Default user prompt includes all sections."""
    user = build_generation_user(_PROBE_STRUCTURE, JD, _PROBE_GAP, original_resume_text="Jane Doe Python SQL")
    assert "resume_structure" in user
    assert "jd_data" in user
    assert "gap_analysis" in user
    assert "candidate_resume_text" in user
    # desired_job_title falls back to JD's role_title.
    assert "Senior Data Engineer" in user


def test_build_generation_user_with_target_title():
    user = build_generation_user(
        _PROBE_STRUCTURE, JD, _PROBE_GAP,
        original_resume_text="Jane Doe", target_title="Staff Engineer",
    )
    assert "Staff Engineer" in user


def test_build_generation_user_with_emphasis():
    user = build_generation_user(
        _PROBE_STRUCTURE, JD, _PROBE_GAP,
        original_resume_text="Jane Doe",
        user_emphasis="leadership experience",
    )
    assert "candidate_emphasis_request" in user
    assert "leadership experience" in user
    assert "EMPHASIZE" in user


def test_build_generation_user_with_repair_instructions():
    user = build_generation_user(
        _PROBE_STRUCTURE, JD, _PROBE_GAP,
        original_resume_text="Jane Doe",
        repair_instructions=["Reduce Python mentions"],
    )
    assert "repair_instructions" in user
    assert "Reduce Python mentions" in user
    assert "- Reduce Python mentions" in user


def test_build_generation_user_without_optional_sections():
    """No target_title default → uses JD title; no emphasis/repair → those sections absent."""
    user = build_generation_user(_PROBE_STRUCTURE, JD, _PROBE_GAP, original_resume_text="Jane Doe")
    assert "candidate_emphasis_request" not in user
    assert "repair_instructions" not in user


# ---------------------------------------------------------------------------
# generate_aligned_content — token-budget selection (gateway vs direct)
# ---------------------------------------------------------------------------
def test_generate_aligned_content_gateway_budget(monkeypatch):
    """On a gateway, max_tokens = GATEWAY_START_TOKENS."""
    monkeypatch.setattr(config, "is_gateway", lambda: True)
    captured = {}

    def fake_call(model=None, system=None, user=None, output_model=None,
                  max_tokens=None, on_token=None, thinking=None):
        captured["max_tokens"] = max_tokens
        captured["thinking"] = thinking
        return _content()

    monkeypatch.setattr("tools.generate_content.structured_call", fake_call)
    generate_aligned_content(_PROBE_STRUCTURE, JD, _PROBE_GAP, original_resume_text="x")
    assert captured["max_tokens"] == config.GATEWAY_START_TOKENS
    # Gateway path: thinking is disabled (gateways auto-think).
    assert captured["thinking"] is False


def test_generate_aligned_content_direct_budget(monkeypatch):
    """On direct API, max_tokens = DIRECT_MAX_TOKENS_GENERATE and thinking=True."""
    monkeypatch.setattr(config, "is_gateway", lambda: False)
    captured = {}

    def fake_call(model=None, system=None, user=None, output_model=None,
                  max_tokens=None, on_token=None, thinking=None):
        captured["max_tokens"] = max_tokens
        captured["thinking"] = thinking
        return _content()

    monkeypatch.setattr("tools.generate_content.structured_call", fake_call)
    generate_aligned_content(_PROBE_STRUCTURE, JD, _PROBE_GAP, original_resume_text="x")
    assert captured["max_tokens"] == config.DIRECT_MAX_TOKENS_GENERATE
    assert captured["thinking"] is True
