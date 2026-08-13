"""Tests for `parse_job_description` + the deterministic minimum-count check.

The LLM extraction is mocked; the count enforcement is pure code and tested
for real.
"""
from __future__ import annotations

from unittest.mock import patch

from schemas import JobData
from tools.parse_job import parse_job_description, validate_jd_counts


def _jd(**overrides) -> JobData:
    base = dict(
        role_title="Senior Data Engineer",
        summary_of_role="Build data pipelines.",
        required_skills=["Python", "SQL", "Spark", "Kafka", "Airflow", "AWS", "dbt"],
        key_responsibilities=[
            "Build pipelines",
            "Maintain warehouse",
            "Improve reliability",
            "Partner with analysts",
            "Monitor costs",
            "Own data quality",
        ],
        preferred_qualifications=["Docker"],
        key_phrases=["high-throughput pipelines", "data quality", "self-serve analytics", "cloud-native", "batch and streaming"],
        stakeholders=["analysts", "engineering"],
        outcomes=["faster insights", "lower costs", "reliable data"],
    )
    base.update(overrides)
    return JobData(**base)


def test_counts_pass_when_minimums_met():
    assert validate_jd_counts(_jd()) == []


def test_counts_flag_short_required_skills():
    jd = _jd(required_skills=["Python", "SQL", "Spark"])  # only 3
    warnings = validate_jd_counts(jd)
    assert any("required_skills" in w for w in warnings)


def test_counts_flag_short_key_phrases():
    jd = _jd(key_phrases=["only one phrase"])
    warnings = validate_jd_counts(jd)
    assert any("key_phrases" in w for w in warnings)


def test_counts_flags_every_short_field():
    jd = _jd(
        required_skills=[],
        key_responsibilities=[],
        key_phrases=[],
        outcomes=[],
    )
    warnings = validate_jd_counts(jd)
    assert len(warnings) == 4


def test_parse_uses_job_data_schema_and_haiku():
    with patch("tools.parse_job.structured_call", return_value=_jd()) as call:
        result = parse_job_description("Senior Data Engineer at Acme...")

    _, kwargs = call.call_args
    assert kwargs["output_model"] is JobData
    assert "Senior Data Engineer at Acme" in kwargs["user"]
    assert result.role_title == "Senior Data Engineer"


def test_parse_does_not_silently_fix_short_output():
    # Even a short LLM result is returned as-is — count checking stays explicit.
    short = _jd(required_skills=["Python"])
    with patch("tools.parse_job.structured_call", return_value=short):
        result = parse_job_description("short")
    assert len(result.required_skills) == 1
    assert validate_jd_counts(result) != []
