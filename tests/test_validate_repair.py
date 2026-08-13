"""Tests for `validate_and_repair` — plumbing + the anti-strip contract.

The LLM call is mocked; what matters is that the corrected output is a
ResumeContent, the correct model/schema is used, and — critically — that the
`intentionally_added` list (bug #5) reaches the prompt so validation never
strips JD coverage that was deliberately added.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import config
from schemas import ContentContact, JobData, ResumeContent
from tools.validate_repair import validate_and_repair

JD = JobData(
    role_title="Data Engineer",
    summary_of_role="Build pipelines.",
    required_skills=["Python", "Kubernetes", "Kafka", "Airflow", "Spark", "SQL"],
    key_responsibilities=["a", "b", "c", "d", "e", "f"],
    key_phrases=["cloud-native", "data quality"],
    outcomes=["reliability", "speed", "costs"],
)

CONTENT = ResumeContent(
    contact=ContentContact(name="Jane", phone="1", email="j@c.com", linkedin="li/j"),
    summary="Engineer.",
)


def test_validate_returns_corrected_content():
    with patch("tools.validate_repair.structured_call", return_value=CONTENT) as call:
        result = validate_and_repair("original resume text", JD, CONTENT)

    assert result is CONTENT
    _, kwargs = call.call_args
    assert kwargs["output_model"] is ResumeContent
    # Gateway-aware budget (GATEWAY_START_TOKENS on a combo gateway, the small
    # direct-API budget otherwise) — assert against the same expression the
    # code uses, not a literal.
    assert kwargs["max_tokens"] == (
        config.GATEWAY_START_TOKENS if config.is_gateway() else config.DIRECT_MAX_TOKENS_VALIDATE
    )


def test_validate_uses_gateway_budget_when_gateway(monkeypatch):
    """On a combo gateway validation starts at GATEWAY_START_TOKENS (64k) — the
    measured happy-path budget where think+emit fit in one attempt. A 16k start
    was burned entirely on auto-thinking and the recovery re-thought from
    scratch, stalling for minutes (verified live)."""
    monkeypatch.setattr(config, "API_BASE_URL", "http://gateway.test")
    with patch("tools.validate_repair.structured_call", return_value=CONTENT) as call:
        validate_and_repair("orig", JD, CONTENT)
    _, kwargs = call.call_args
    assert kwargs["max_tokens"] == config.GATEWAY_START_TOKENS


def test_validate_uses_configured_model():
    with patch("tools.validate_repair.structured_call", return_value=CONTENT) as call:
        validate_and_repair("orig", JD, CONTENT)
    _, kwargs = call.call_args
    # The validate stage is the cheap pass (Haiku on the direct API, whatever
    # MODEL_VALIDATE resolves to on a gateway) — assert against config, not a
    # literal model ID, so the suite is environment-agnostic.
    assert kwargs["model"] == config.MODEL_VALIDATE


def test_original_text_and_generated_content_reach_prompt():
    with patch("tools.validate_repair.structured_call", return_value=CONTENT) as call:
        validate_and_repair("GROUND TRUTH RESUME", JD, CONTENT)

    _, kwargs = call.call_args
    user = kwargs["user"]
    assert "GROUND TRUTH RESUME" in user
    assert "ORIGINAL_RESUME_TEXT" in user
    assert "GENERATED_RESUME_JSON" in user
    assert json.loads(user.split("## JOB_DESCRIPTION_DATA")[1].split("## GENERATED_RESUME_JSON")[0].strip())


def test_intentionally_added_reaches_prompt_as_do_not_strip():
    with patch("tools.validate_repair.structured_call", return_value=CONTENT) as call:
        validate_and_repair(
            "orig", JD, CONTENT, intentionally_added=["Kubernetes", "data quality"]
        )

    _, kwargs = call.call_args
    user = kwargs["user"]
    assert "## INTENTIONALLY_ADDED" in user
    assert "Kubernetes" in user
    assert "data quality" in user
    # The anti-strip rule is in the system prompt verbatim.
    assert "NOT a hallucination" in kwargs["system"]
    assert "Do NOT remove, flag, or 'correct' content just because it isn't found" in kwargs["system"]


def test_validate_emits_schema_not_report():
    # The system prompt must demand corrected JSON, never prose.
    with patch("tools.validate_repair.structured_call", return_value=CONTENT) as call:
        validate_and_repair("orig", JD, CONTENT)
    _, kwargs = call.call_args
    assert "Output ONLY the corrected resume JSON" in kwargs["system"]
