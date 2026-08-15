"""LangGraph node callables — each wraps one tool and streams live events.

Every node:
  1. reads its inputs from `state`,
  2. calls the underlying tool (deterministic guardrail or LLM),
  3. writes its result back to `state`,
  4. appends a human-readable entry to `state["step_log"]`,
  5. emits live events through the injected `writer` so the UI can stream the
     agent's reasoning in real time (LangGraph `stream_mode="custom"`).

LLM stages additionally stream raw tokens through the writer so the UI shows
live progress during the long Sonnet generation call.

Exceptions are deliberately NOT swallowed: they propagate out of the graph so
the caller (the Streamlit UI) surfaces the failure honestly.
"""
from __future__ import annotations

import re
from typing import Any

from langgraph.types import StreamWriter

from graph.state import ResumeAgentState
from rendering.html_renderer import build_full_html
from rendering.pdf_renderer import render_pdf
from tools.alignment_gap import check_alignment_gap
from tools.generate_content import (
    generate_aligned_content,
    run_post_generation_checks,
    summarize_differences,
)
from tools.parse_job import parse_job_description
from tools.parse_resume import parse_resume_structure
from tools.validate_repair import validate_and_repair

NodeOutput = dict[str, Any]


def _step(label: str, status: str, detail: str = "") -> dict:
    entry: dict[str, str] = {"label": label, "status": status}
    if detail:
        entry["detail"] = detail
    return entry


def _emit(writer: StreamWriter, event: dict) -> None:
    writer(event)


# ---------------------------------------------------------------------------
# Parse stage
# ---------------------------------------------------------------------------
def parse_resume_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    label = "Parse resume structure"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    structure = parse_resume_structure(
        state["resume_text"],
        on_token=lambda t: _emit(writer, {"type": "token", "text": t}),
    )
    detail = (
        f"{structure.employer_count} employer(s), "
        f"seniority {structure.estimated_seniority}"
    )
    _emit(writer, {"type": "step", "label": label, "status": "done", "detail": detail})
    return {"resume_structure": structure, "step_log": [_step(label, "done", detail)]}


def parse_job_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    label = "Parse job description"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    jd = parse_job_description(
        state["jd_text"],
        on_token=lambda t: _emit(writer, {"type": "token", "text": t}),
    )
    detail = f"Role: {jd.role_title} | {len(jd.required_skills)} required skills"
    _emit(writer, {"type": "step", "label": label, "status": "done", "detail": detail})
    return {"jd_data": jd, "step_log": [_step(label, "done", detail)]}


# ---------------------------------------------------------------------------
# Deterministic alignment-gap analysis (no LLM)
# ---------------------------------------------------------------------------
def check_gap_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    label = "Analyze alignment gap"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    gap = check_alignment_gap(
        state["resume_text"], state["resume_structure"], state["jd_data"]
    )
    _emit(writer, {"type": "reasoning", "text": gap.reasoning})
    _emit(
        writer,
        {"type": "step", "label": label, "status": "done", "detail": gap.reasoning},
    )
    return {"gap_analysis": gap, "step_log": [_step(label, "done", gap.reasoning)]}


# ---------------------------------------------------------------------------
# Generation + deterministic post-checks
# ---------------------------------------------------------------------------
def generate_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    repair = list(state.get("repair_instructions") or [])
    pass_no = state.get("validation_attempts", 0) + 1  # 1-based generation pass
    label = "Generate tailored resume" if pass_no == 1 else f"Regenerate (pass {pass_no})"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    if repair:
        note = f"Applying {len(repair)} repair instruction(s)"
        _emit(writer, {"type": "step", "label": label, "status": "running", "detail": note})

    gap = state["gap_analysis"]
    content = generate_aligned_content(
        state["resume_structure"],
        state["jd_data"],
        gap,
        original_resume_text=state["resume_text"],
        target_title=state.get("target_title"),
        user_emphasis=state.get("user_emphasis"),
        repair_instructions=repair or None,
        on_token=lambda t: _emit(writer, {"type": "token", "text": t}),
    )

    # Deterministic post-generation checks run immediately (bugs #1/#3/#4/#6/#7/#9).
    content, report = run_post_generation_checks(
        content,
        state["jd_data"],
        state["resume_text"],
        gap.depth_target,
        resume_structure=state["resume_structure"],
    )
    verdict = "passed" if report.passed else f"{len(report.issues)} issue(s) found"
    _emit(writer, {"type": "step", "label": label, "status": "done", "detail": f"Post-checks {verdict}"})

    entries = [_step(label, "done", f"Post-checks {verdict}")]
    entries += [_step("Post-check issue", "running", i.message) for i in report.issues]
    return {
        "generated_content": content,
        "postcheck_report": report,
        "repair_instructions": report.repair_instructions,
        "step_log": entries,
    }


# ---------------------------------------------------------------------------
# Validation & repair (second, cheaper LLM pass)
# ---------------------------------------------------------------------------
def validate_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    label = "Validate & repair"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    gap = state["gap_analysis"]
    # Bug #5: JD content deliberately added must never be stripped as a hallucination.
    intentionally_added = list(gap.missing_skills) + list(gap.missing_phrases)
    repaired = validate_and_repair(
        state["resume_text"],
        state["jd_data"],
        state["generated_content"],
        intentionally_added=intentionally_added,
        on_token=lambda t: _emit(writer, {"type": "token", "text": t}),
    )

    # The validate pass is another LLM and can RE-INTRODUCE code-enforced
    # guardrail violations (verified live: it re-saturated Python/SQL past the
    # keyword cap right after the generation pass had passed post-checks). Guard
    # its output with the SAME deterministic checks, and let the repair loop see
    # them — otherwise a regression after the final post-check ships unguarded.
    repaired, report = run_post_generation_checks(
        repaired,
        state["jd_data"],
        state["resume_text"],
        gap.depth_target,
        resume_structure=state["resume_structure"],
    )

    # Regression guard: if validation made a guardrail-clean generation worse,
    # keep the clean generation — the candidate gets a document that satisfies
    # every code-enforced rule (JD coverage is already satisfied, since the
    # generation passed all post-checks).
    prev_report = state.get("postcheck_report")
    if report.issues and prev_report is not None and prev_report.passed:
        repaired = state["generated_content"]
        report = prev_report

    attempts = state.get("validation_attempts", 0) + 1
    verdict = "passed" if report.passed else f"{len(report.issues)} issue(s) found"
    detail = f"Pass {attempts} · post-checks {verdict}"
    _emit(writer, {"type": "step", "label": label, "status": "done", "detail": detail})

    entries = [_step(label, "done", detail)]
    entries += [_step("Validate post-check issue", "running", i.message) for i in report.issues]
    return {
        "repaired_content": repaired,
        "validation_attempts": attempts,
        "postcheck_report": report,
        "repair_instructions": report.repair_instructions,
        "step_log": entries,
    }


# ---------------------------------------------------------------------------
# Deterministic rendering (HTML + PDF) — no LLM in the output path
# ---------------------------------------------------------------------------
def render_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    content = state.get("repaired_content") or state.get("generated_content")
    structure = state["resume_structure"]
    # Header subtitle under the candidate's name (n8n's form "Job Role" field) —
    # the user-supplied target title, falling back to the JD's own role title.
    title = state.get("target_title") or state["jd_data"].role_title

    label = "Render HTML"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    html_doc = build_full_html(content, structure, title)
    _emit(writer, {"type": "step", "label": label, "status": "done", "detail": f"{len(html_doc):,} chars"})

    label = "Render PDF"
    _emit(writer, {"type": "step", "label": label, "status": "running"})
    pdf_error: str | None = None
    step_log: list = []
    try:
        pdf_bytes = render_pdf(html_doc)
        _emit(writer, {"type": "step", "label": label, "status": "done", "detail": f"{len(pdf_bytes):,} bytes"})
        step_log = [
            _step("Render HTML", "done", f"{len(html_doc):,} chars"),
            _step("Render PDF", "done", f"{len(pdf_bytes):,} bytes"),
        ]
    except RuntimeError as exc:
        # No PDF backend is available — neither Playwright's Chromium (e.g.
        # Streamlit Community Cloud, which doesn't install Playwright browsers)
        # nor weasyprint. The tailored resume is still complete as HTML —
        # surface the PDF gap to the UI rather than failing the whole run.
        _emit(writer, {"type": "step", "label": label, "status": "failed", "detail": str(exc)})
        pdf_bytes = None
        pdf_error = str(exc)
        step_log = [
            _step("Render HTML", "done", f"{len(html_doc):,} chars"),
            _step("Render PDF", "failed", pdf_error),
        ]

    return {
        "html": html_doc,
        "pdf_bytes": pdf_bytes,
        "pdf_render_error": pdf_error,
        "step_log": step_log,
    }


# ---------------------------------------------------------------------------
# Finalize: diff summary + safe download filename
# ---------------------------------------------------------------------------
def finalize_node(state: ResumeAgentState, writer: StreamWriter) -> NodeOutput:
    content = state.get("repaired_content") or state.get("generated_content")
    jd = state["jd_data"]
    summary = summarize_differences(content, jd)
    title = state.get("target_title") or jd.role_title
    slug = re.sub(r"[^\w\-]+", "_", title.strip(), flags=re.UNICODE)
    file_name = re.sub(r"_+", "_", slug).strip("_")
    detail = (
        f"Done. {len(summary.get('added_skills', []))}/{summary.get('total_required', 0)} "
        "JD required skills now present."
    )
    _emit(writer, {"type": "step", "label": "Finalize", "status": "done", "detail": detail})
    return {
        "final_summary": summary,
        "final_file_name": file_name,
        "step_log": [_step("Finalize", "done", detail)],
    }
