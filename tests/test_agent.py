"""Tests for the LangGraph agent wiring — control flow, repair loop, streaming.

All LLM tools are mocked; the deterministic stages (gap analysis, post-checks,
HTML rendering) run for real, and `render_pdf` is mocked so no browser launches.
These tests prove the agent's *shape*: node order, bounded repair loop,
repair-instruction + intentionally-added propagation, and live event streaming.
"""
from __future__ import annotations

import contextlib

import pytest
from unittest import mock
from unittest.mock import patch

import config
from graph.agent import build_agent, should_repair
from graph.stream import build_input, run_agent
from schemas import (
    ContentContact,
    ContentEmployer,
    Education,
    EmployerProbe,
    Issue,
    JobData,
    PostCheckReport,
    ResumeContent,
    ResumeStructure,
    SkillCategory,
)

ORIGINAL = "Jane Doe Acme Corp 2019-2020 Python SQL Spark pipelines data quality analysts"

STRUCTURE = ResumeStructure(
    sections_present=["PROFESSIONAL SUMMARY", "WORK EXPERIENCE", "SKILLS", "EDUCATION"],
    employers=[
        EmployerProbe(
            name="Acme Corp", has_clients=False, clients=[], start_date="2019", end_date="2020"
        )
    ],
    employer_count=1,
    estimated_seniority="mid",
    contact_fields_found=["name", "phone", "email", "linkedin"],
)

JD = JobData(
    role_title="Data Engineer",
    summary_of_role="Build and run data pipelines.",
    required_skills=["Python", "SQL", "Spark", "Kubernetes", "Kafka", "Airflow"],
    key_responsibilities=["a", "b", "c", "d", "e", "f"],
    key_phrases=["cloud-native", "data quality"],
    preferred_qualifications=["GCP"],
    stakeholders=["analysts"],
    outcomes=["reliability", "lower costs", "faster delivery"],
)


def passing_content() -> ResumeContent:
    """Passes every deterministic post-check (mid seniority, 1 employer, floor 4)."""
    return ResumeContent(
        contact=ContentContact(name="Jane Doe", phone="555", email="j@c.com", linkedin="li/j"),
        summary="Engineer building Python and SQL pipelines with Spark.",
        employers=[
            ContentEmployer(
                name="Acme Corp",
                start_date="2019",
                end_date="2020",
                role_title="Data Engineer",
                description="Built the data platform.",
                has_clients=False,
                bullets=[
                    "Wrote Python ETL jobs.",
                    "Optimized SQL warehouse queries.",
                    "Tuned Spark cluster memory.",
                    "Partnered with analysts on reliability.",
                    "Cut storage costs 20%.",
                ],
            )
        ],
        skills=[SkillCategory(category="Core", items=["Python", "SQL", "Spark"])],
        education=[Education(institution="MIT", degree="BS", year="2015")],
    )


def failing_content() -> ResumeContent:
    """Python appears 6x (4 in summary) — trips the keyword_cap guardrail."""
    c = passing_content()
    c.summary = "Python expert. Python architect. Python lead. Python fanatic."
    return c


def _patch_tools(
    *,
    generate=None,
    generate_side_effect=None,
    validate=None,
    validate_side_effect=None,
):
    """Patch every LLM tool + the PDF renderer inside graph.nodes.

    Holds explicit references to the Mock objects (patch.multiple's returned
    dict is empty on this Python), returned as a dict keyed by attribute name.
    """
    gen = (
        mock.Mock(side_effect=generate_side_effect)
        if generate_side_effect is not None
        else mock.Mock(return_value=generate or passing_content())
    )
    val = (
        mock.Mock(side_effect=validate_side_effect)
        if validate_side_effect is not None
        else mock.Mock(
            return_value=validate
            or passing_content().model_copy(update={"summary": "MARKER validated summary."})
        )
    )
    mocks = {
        "parse_resume_structure": mock.Mock(return_value=STRUCTURE),
        "parse_job_description": mock.Mock(return_value=JD),
        "generate_aligned_content": gen,
        "validate_and_repair": val,
        "render_pdf": mock.Mock(return_value=b"%PDF-fake"),
    }
    patcher = patch.multiple("graph.nodes", **mocks)
    patcher.start()
    return patcher, mocks


@contextlib.contextmanager
def _patched(**kwargs):
    patcher, mocks = _patch_tools(**kwargs)
    try:
        yield mocks
    finally:
        patcher.stop()


def test_happy_path_runs_end_to_end():
    with _patched() as mocks:
        state = build_agent().invoke(build_input(ORIGINAL, "JD text here"))

    # Parsed + analyzed.
    assert state["resume_structure"] is STRUCTURE
    assert state["jd_data"] is JD
    assert state["gap_analysis"].severity == "heavy_tailoring"

    # Generated + validated once (clean report → no repair loop).
    assert mocks["generate_aligned_content"].call_count == 1
    assert mocks["validate_and_repair"].call_count == 1
    assert state["validation_attempts"] == 1
    assert state["postcheck_report"].passed is True
    assert state["generated_content"] is mocks["generate_aligned_content"].return_value

    # Render uses the VALIDATED content (marker proves it), not the raw generation.
    assert "MARKER" in state["html"]
    assert "<!DOCTYPE html>" in state["html"]
    assert '<meta charset="utf-8">' in state["html"]
    assert state["pdf_bytes"] == b"%PDF-fake"

    # Finalize.
    assert state["final_file_name"] == "Data_Engineer"
    assert state["final_summary"]["total_required"] == 6
    assert state["final_summary"]["added_skills"] == ["Python", "SQL", "Spark"]

    # Transcript: first entry is the parse, last is finalize, in order.
    labels = [e["label"] for e in state["step_log"]]
    assert labels[0] == "Parse resume structure"
    assert labels[-1] == "Finalize"
    assert "Analyze alignment gap" in labels
    assert "Render PDF" in labels


def test_validation_receives_intentionally_added():
    """Bug #5: validation must be told exactly what was deliberately added."""
    with _patched() as mocks:
        build_agent().invoke(build_input(ORIGINAL, "JD text here"))
    kwargs = mocks["validate_and_repair"].call_args.kwargs
    assert kwargs["intentionally_added"] == ["Kubernetes", "Kafka", "Airflow", "cloud-native"]


def test_user_options_propagate_to_generation():
    with _patched() as mocks:
        state = build_agent().invoke(
            build_input(
                ORIGINAL, "JD text here", user_emphasis="leadership", target_title="Staff Data Engineer"
            )
        )
    kw = mocks["generate_aligned_content"].call_args.kwargs
    assert kw["target_title"] == "Staff Data Engineer"
    assert kw["user_emphasis"] == "leadership"
    assert state["final_file_name"] == "Staff_Data_Engineer"


def test_repair_loop_is_bounded_and_still_renders():
    """A permanently-failing postcheck must not loop forever: bounded at
    MAX_REPAIR_ITERATIONS, then render whatever we have (honest output)."""
    with _patched(
        generate=failing_content(), validate_side_effect=lambda *a, **k: a[2]
    ) as mocks:
        state = build_agent().invoke(build_input(ORIGINAL, "JD text here"))

    assert mocks["generate_aligned_content"].call_count == config.MAX_REPAIR_ITERATIONS + 1
    assert mocks["validate_and_repair"].call_count == config.MAX_REPAIR_ITERATIONS + 1
    assert state["validation_attempts"] == config.MAX_REPAIR_ITERATIONS + 1
    assert state["postcheck_report"].passed is False  # never fixed — but bounded
    assert state["pdf_bytes"] == b"%PDF-fake"          # rendered anyway


def test_repair_instructions_feed_the_next_pass():
    """The deterministic post-check issues must reach the 2nd generation call."""
    with _patched(
        generate=failing_content(), validate_side_effect=lambda *a, **k: a[2]
    ) as mocks:
        build_agent().invoke(build_input(ORIGINAL, "JD text here"))

    calls = mocks["generate_aligned_content"].call_args_list
    assert calls[0].kwargs["repair_instructions"] in (None, [])   # fresh start
    assert calls[1].kwargs["repair_instructions"]                  # fed forward
    assert any("Python" in r for r in calls[1].kwargs["repair_instructions"])


def test_validate_regression_keeps_clean_generation():
    """If the validate pass makes a guardrail-clean generation WORSE (bug #4:
    re-saturating a keyword past the cap right after generation passed
    post-checks), the clean generation must be the final content — validation
    must not ship an unguarded regression."""
    with _patched(generate=passing_content(), validate=failing_content()) as mocks:
        state = build_agent().invoke(build_input(ORIGINAL, "JD text here"))

    assert mocks["generate_aligned_content"].call_count == 1  # clean → no repair loop
    assert mocks["validate_and_repair"].call_count == 1
    assert state["validation_attempts"] == 1
    assert state["postcheck_report"].passed is True  # regression guard kept the clean report
    # Final content is the clean generation, not the violating validation output.
    assert state["repaired_content"] is state["generated_content"]
    assert "Python expert." not in state["repaired_content"].summary
    assert "Python expert." not in state["html"]


def test_should_repair_edge():
    def state(report, attempts):
        return {"postcheck_report": report, "validation_attempts": attempts}

    clean = PostCheckReport(passed=True, issues=[])
    # Clean report → render.
    assert should_repair(state(clean, 0)) == "render"

    # Failing with concrete repair instructions, within budget → repair.
    failing = PostCheckReport(
        passed=False, issues=[Issue(code="keyword_cap", message="m", repair="Fix it.")]
    )
    assert should_repair(state(failing, 1)) == "repair"

    # Budget exhausted → render (bounded).
    assert (
        should_repair(state(failing, config.MAX_REPAIR_ITERATIONS + 1)) == "render"
    )

    # Failing but nothing to repair (already auto-fixed) → render.
    nofix = PostCheckReport(
        passed=False, issues=[Issue(code="client_grounding", message="m", repair="")]
    )
    assert should_repair(state(nofix, 0)) == "render"


def _streaming_generate(*args, **kwargs):
    """Simulate the real generator: it calls on_token while producing output."""
    kwargs["on_token"]("TOK1")
    kwargs["on_token"]("TOK2")
    return passing_content()


def test_run_agent_streams_live_events():
    with _patched(generate_side_effect=_streaming_generate):
        events = list(run_agent(build_input(ORIGINAL, "JD text here")))

    types = [e["type"] for e in events]
    assert "step" in types
    assert "reasoning" in types          # gap analysis reasoning emitted live
    assert "token" in types              # generation tokens streamed via on_token
    assert "state" in types

    # The FINAL state snapshot carries the renderable outputs.
    final = next(e for e in reversed(events) if e["type"] == "state")
    assert final["state"]["pdf_bytes"] == b"%PDF-fake"
    assert final["state"]["final_file_name"] == "Data_Engineer"

    # Steps arrive in the agent's working order (running → done).
    steps = [e for e in events if e["type"] == "step"]
    assert steps[0]["label"] == "Parse resume structure"
    assert steps[0]["status"] == "running"


def test_build_input_requires_text():
    with pytest.raises(ValueError):
        build_input("", "JD text here")
    with pytest.raises(ValueError):
        build_input("resume text", "   ")
