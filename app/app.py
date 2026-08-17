"""Streamlit UI for the Resume Optimizer agent.

Run from the project root:

    streamlit run app/app.py

Upload (or paste) a resume, paste a job description, optionally add an emphasis
request or a target job title, then click "Tailor resume". The agent's live
reasoning steps stream into the page; when it finishes you get a before/after
summary, an HTML preview, and a downloadable PDF.

The PDF is rendered locally via Playwright — nothing leaves the machine except
the LLM calls to the Anthropic API (key from `.env`).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Make `config`, `graph`, `tools`, `rendering` importable no matter where
# streamlit was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pypdf import PdfReader

import config
from graph.stream import build_input, run_agent

st.set_page_config(
    page_title="Resume Optimizer",
    page_icon=":material/description:",
    layout="wide",
)

if not config.ANTHROPIC_API_KEY:
    st.error(
        "`ANTHROPIC_API_KEY` is not set. Copy `.env.example` to `.env`, fill in "
        "your key, then restart the app.",
        icon=":material/error:",
    )
    st.stop()

with st.sidebar:
    st.markdown("### :material/description: Resume Optimizer")
    st.caption("LangGraph agent · Claude · Playwright")
    with st.expander("How it works", icon=":material/info:"):
        st.markdown(
            "1. Parse the resume and job description\n"
            "2. Analyze the alignment gap\n"
            "3. Generate tailored content\n"
            "4. Validate and repair (bounded loop)\n"
            "5. Render HTML → PDF"
        )
    st.caption("Runs locally — only LLM calls leave the machine.")

st.title("Resume Optimizer")
st.caption(
    "Tailors a resume to a job description: parses both, analyzes the gap, "
    "generates aligned content, validates it, and renders a PDF."
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col_resume, col_jd = st.columns(2)

with col_resume, st.container(border=True):
    st.subheader(":material/upload_file: Resume")
    source = st.segmented_control(
        "Source", ["Upload PDF", "Paste text"], default="Upload PDF"
    )
    resume_text = ""
    if source == "Upload PDF":
        uploaded = st.file_uploader("Resume PDF", type=["pdf"])
        if uploaded is not None:
            try:
                reader = PdfReader(io.BytesIO(uploaded.getvalue()))
                resume_text = "\n".join((p.extract_text() or "") for p in reader.pages)
                st.success(
                    f"Extracted {len(resume_text):,} chars from {len(reader.pages)} page(s).",
                    icon=":material/check_circle:",
                )
            except Exception as exc:  # pypdf chokes on corrupt/garbled PDFs
                st.error(f"Could not read that PDF: {exc}", icon=":material/error:")
    else:
        resume_text = st.text_area("Resume text", height=260, key="resume_text_input")

with col_jd, st.container(border=True):
    st.subheader(":material/work: Job description")
    jd_text = st.text_area(
        "Paste the full job posting",
        height=260,
        key="jd_text_input",
    )

with st.container(border=True):
    st.subheader(":material/tune: Optional adjustments")
    o1, o2 = st.columns(2)
    with o1:
        target_title = st.text_input(
            "Target job title", placeholder="e.g. Staff Data Engineer"
        )
    with o2:
        user_emphasis = st.text_input(
            "Emphasis request", placeholder="e.g. highlight leadership"
        )

ready = bool(resume_text.strip() and jd_text.strip())
if not ready:
    st.caption("Provide a resume and a job description above, then run the agent.")
go = st.button(
    "Tailor resume",
    type="primary",
    icon=":material/auto_awesome:",
    disabled=not ready,
    width="stretch",
)

SEVERITY_BADGE = {
    "full_rebuild": ("red", ":material/build:"),
    "heavy_tailoring": ("orange", ":material/edit:"),
    "light_tweak": ("green", ":material/check:"),
}


def render_results(state: dict) -> None:
    """Before/after summary, HTML preview, and PDF download."""
    st.subheader(":material/task_alt: Result")

    summary = state.get("final_summary") or {}
    added = summary.get("added_skills", [])
    total = summary.get("total_required", 0)
    gap = state.get("gap_analysis")

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "JD required skills present",
            f"{len(added)}/{total}",
            icon=":material/verified:",
            border=True,
        )
        m2.metric(
            "Generation/validation passes",
            state.get("validation_attempts", 0),
            icon=":material/sync:",
            border=True,
        )
        m3.metric(
            "PDF size",
            f"{len(state.get('pdf_bytes') or b'')/1024:.1f} KB",
            icon=":material/picture_as_pdf:",
            border=True,
        )

    if gap is not None:
        color, icon = SEVERITY_BADGE.get(gap.severity, ("gray", ":material/help:"))
        with st.container(border=True):
            st.badge(
                gap.severity.replace("_", " ").title(),
                icon=icon,
                color=color,
            )
            st.write(gap.reasoning)

    st.write("**Added to the tailored resume (was missing from the source):**")
    if added:
        st.markdown(" ".join(f":blue-badge[{skill}]" for skill in added))
    else:
        st.caption("None — the source already covered every required skill.")

    pdf_bytes = state.get("pdf_bytes")
    pdf_error = state.get("pdf_render_error")
    if pdf_bytes:
        st.download_button(
            "Download tailored PDF",
            data=pdf_bytes,
            file_name=f"{state.get('final_file_name', 'resume')}.pdf",
            mime="application/pdf",
            type="primary",
            icon=":material/download:",
        )
    elif pdf_error:
        st.warning(
            pdf_error
            + "\n\nTo enable PDF downloads, install a Chromium-based browser "
            "(Playwright Chromium/Edge/Chrome) OR the `weasyprint` Python package "
            "in this environment (Streamlit Community Cloud can install weasyprint "
            "via requirements.txt + system libs via packages.txt at the repo root), or "
            "run the app locally.",
            icon=":material/warning:",
        )
    with st.expander("Preview rendered HTML", icon=":material/visibility:"):
        st.iframe(state.get("html") or "", height=700)


# ---------------------------------------------------------------------------
# Agent run — live streaming transcript
# ---------------------------------------------------------------------------
if go:
    try:
        inputs = build_input(
            resume_text,
            jd_text,
            user_emphasis=user_emphasis.strip() or None,
            target_title=target_title.strip() or None,
        )
    except ValueError as exc:
        st.warning(str(exc), icon=":material/warning:")
    else:
        st.space("large")
        last_state = None
        token_buf: list[str] = []

        def flush_tokens():
            if token_buf:
                st.code("".join(token_buf), language=None)
                token_buf.clear()

        with st.status("Running the agent…", expanded=True) as status:
            try:
                for event in run_agent(inputs):
                    etype = event.get("type")
                    if etype == "step":
                        line = f"**{event.get('label', '')}** · {event.get('status')}"
                        detail = event.get("detail")
                        if detail:
                            line += f"  \n{detail}"
                        st.markdown(line)
                    elif etype == "reasoning":
                        st.info(event["text"], icon=":material/psychology:")
                    elif etype == "token":
                        token_buf.append(event["text"])
                        if len(token_buf) >= 300:
                            flush_tokens()
                    elif etype == "state":
                        last_state = event["state"]
            except Exception as exc:
                st.error(f"The agent failed: {exc}", icon=":material/error:")
                status.update(label="Failed", state="error")
            else:
                flush_tokens()
                if last_state is not None:
                    status.update(
                        label=f"Done — {last_state.get('final_file_name', 'resume')} ready",
                        state="complete",
                    )
                else:
                    status.update(label="Failed", state="error")

        if last_state is not None:
            render_results(last_state)
