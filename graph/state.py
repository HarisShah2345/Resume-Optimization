"""Typed agent state shared by every LangGraph node.

The graph is a DAG of tool calls; the state carries the inputs plus the output
of each stage so the UI can render a live transcript and the final result
(HTML, PDF, diff summary) without re-deriving anything.

`step_log` uses an `operator.add` reducer so each node APPENDS its transcript
entries; everything else is plain assignment (last node wins).
"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from schemas import (
    GapAnalysis,
    JobData,
    PostCheckReport,
    ResumeContent,
    ResumeStructure,
)


class ResumeAgentState(TypedDict, total=False):
    """Everything the agent knows across a single run.

    Inputs first, then the parsed/analyzed intermediates, then the generated
    + validated content, then the transcript, then the renderable outputs.
    """

    # --- inputs ---------------------------------------------------------
    resume_text: str
    jd_text: str
    user_emphasis: str | None
    target_title: str | None

    # --- parsed / analyzed ---------------------------------------------
    resume_structure: ResumeStructure | None
    jd_data: JobData | None
    gap_analysis: GapAnalysis | None

    # --- generated & validated content ---------------------------------
    generated_content: ResumeContent | None
    repaired_content: ResumeContent | None
    postcheck_report: PostCheckReport | None
    # Fed into the next generation pass (produced by the deterministic postchecks).
    repair_instructions: list[str]
    # Completed validate passes — bounds the repair loop.
    validation_attempts: int

    # --- transcript -----------------------------------------------------
    step_log: Annotated[list[dict], add]

    # --- outputs --------------------------------------------------------
    html: str | None
    pdf_bytes: bytes | None
    pdf_render_error: str | None
    final_file_name: str | None
    final_summary: dict | None
