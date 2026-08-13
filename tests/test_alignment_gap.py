"""Tests for the deterministic alignment-gap analysis."""
from __future__ import annotations

from schemas import JobData, ResumeStructure
from tools.alignment_gap import check_alignment_gap

RESUME = (
    "Jane Doe\nProfessional Summary\nWork Experience\n"
    "Acme Corp 2019-Present: Built Python services and SQL pipelines. "
    "Managed AWS infrastructure with Kafka and Airflow. Improved reliability. "
    "Reduced monthly cloud costs by 20%. Partnered with data analysts to "
    "deliver self-serve analytics dashboards.\n"
)

REQUIRED = ["Python", "SQL", "Spark", "Kafka", "Airflow", "AWS", "dbt", "Kubernetes"]


def _jd(**kw) -> JobData:
    base = dict(
        role_title="Senior Data Engineer",
        summary_of_role="Build data pipelines.",
        required_skills=list(REQUIRED),
        key_responsibilities=["a", "b", "c", "d", "e", "f"],
        key_phrases=["self-serve analytics", "cloud-native", "high-throughput pipelines", "data quality"],
        outcomes=["reduce monthly cloud costs", "improve pipeline reliability", "launch products for internal teams"],
    )
    base.update(kw)
    return JobData(**base)


def _structure(seniority="mid") -> ResumeStructure:
    return ResumeStructure(
        sections_present=["PROFESSIONAL SUMMARY", "WORK EXPERIENCE"],
        employer_count=1,
        estimated_seniority=seniority,
        contact_fields_found=["name"],
    )


def test_full_rebuild_when_most_skills_missing():
    gap = check_alignment_gap(RESUME, _structure(), _jd())
    # RESUME covers Python, SQL, Kafka, Airflow, AWS = 5/8 -> 62% -> heavy.
    assert gap.severity in ("heavy_tailoring", "full_rebuild")
    assert "Spark" in gap.missing_skills
    assert "Kubernetes" in gap.missing_skills
    assert "dbt" in gap.missing_skills


def test_light_tweak_when_strong_fit():
    jd = _jd(required_skills=["Python", "SQL", "AWS", "Airflow"])
    gap = check_alignment_gap(RESUME, _structure(), jd)
    assert gap.severity == "light_tweak"
    assert gap.missing_skills == []


def test_full_rebuild_when_coverage_very_low():
    jd = _jd(required_skills=["Spark", "Kubernetes", "dbt", "Terraform", "GCP", "Flink"])
    gap = check_alignment_gap(RESUME, _structure(), jd)
    assert gap.severity == "full_rebuild"


def test_missing_phrases_are_verbatim_absent():
    gap = check_alignment_gap(RESUME, _structure(), _jd())
    assert "self-serve analytics" not in gap.missing_phrases  # present verbatim
    assert "cloud-native" in gap.missing_phrases
    assert "high-throughput pipelines" in gap.missing_phrases


def test_missing_outcomes_detected_by_content_words():
    gap = check_alignment_gap(RESUME, _structure(), _jd())
    assert "reduce monthly cloud costs" not in gap.missing_outcomes  # evidence exists
    assert "launch products for internal teams" in gap.missing_outcomes


def test_depth_target_flows_from_resume_seniority():
    gap = check_alignment_gap(RESUME, _structure(seniority="senior"), _jd())
    assert gap.depth_target == "senior"


def test_reasoning_string_is_populated():
    gap = check_alignment_gap(RESUME, _structure(), _jd())
    assert "Severity:" in gap.reasoning
    assert "depth target" in gap.reasoning
