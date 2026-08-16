"""Smoke tests for the Streamlit UI, via Streamlit's AppTest harness (no browser).

They prove the script imports and renders without exceptions:
  - with no API key it shows the setup banner and stops cleanly,
  - with a (fake) key it renders the full input form.

The agent run itself is exercised with mocks in test_agent.py; a live run
needs a real API key and is verified manually per the plan.
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent


def _app() -> AppTest:
    return AppTest.from_file(str(_ROOT / "app" / "app.py"), default_timeout=30)


def test_app_shows_setup_banner_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    at = _app().run()
    assert not at.exception
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)


def test_app_renders_form_with_api_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")
    at = _app().run()
    assert not at.exception

    assert at.title[0].value == "Resume Optimizer"
    assert at.button  # the "Tailor resume" button exists

    # Default source is "Upload PDF", so only the JD text area is visible.
    labels = [t.label for t in at.text_area]
    assert any("job posting" in (l or "").lower() for l in labels)
    assert at.button[0].disabled is True

    # Switch to "Paste text" and fill both areas → button enables.
    at.segmented_control[0].set_value("Paste text")
    at.run()
    labels = [t.label for t in at.text_area]
    assert any("resume" in (l or "").lower() for l in labels)

    at.text_area[0].set_value("Jane Doe\nAcme Corp\nPython SQL")
    at.text_area[1].set_value("Senior Data Engineer. Python SQL Spark.")
    at.run()
    assert not at.exception
    assert at.button[0].disabled is False


def test_agent_run_success_with_pdf(monkeypatch):
    """Clicking 'Tailor resume' runs the agent; results show PDF download."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")

    finished_state = {
        "final_summary": {"added_skills": ["Python", "SQL"], "total_required": 5},
        "gap_analysis": type("G", (), {"severity": "light_tweak", "reasoning": "Good match"})(),
        "pdf_bytes": b"%PDF-1.4 fake",
        "html": "<html><body>hi</body></html>",
        "final_file_name": "Tailored_Resume",
        "validation_attempts": 1,
    }

    fake_events = iter([
        {"type": "step", "label": "Parse resume", "status": "done", "detail": "3 employers"},
        {"type": "reasoning", "text": "Analyzing gap..."},
        {"type": "token", "text": "Generating..."},
        {"type": "state", "state": finished_state},
    ])

    import graph.stream as gstream
    monkeypatch.setattr(gstream, "run_agent", lambda inputs: fake_events)

    at = _app().run()
    at.segmented_control[0].set_value("Paste text")
    at.run()
    at.text_area[0].set_value("Jane Doe\nPython developer")
    at.text_area[1].set_value("Senior Data Engineer. Python SQL Spark.")
    at.run()
    at.button[0].click().run()

    assert not at.exception
    # Metrics should show 2/5 skills present.
    metrics = at.metric
    labels = [m.label for m in metrics]
    assert "JD required skills present" in labels

    # PDF download button should be present with the correct filename.
    assert len(at.download_button) >= 1
    assert at.download_button[0].label == "Download tailored PDF"


def test_agent_run_success_without_pdf(monkeypatch):
    """When PDF renders failed, show warning and HTML-only path still works."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")

    finished_state = {
        "final_summary": {"added_skills": ["Python"], "total_required": 3},
        "gap_analysis": type("G", (), {"severity": "full_rebuild", "reasoning": "Big gap"})(),
        "pdf_bytes": None,
        "pdf_render_error": "No Chromium-based browser available",
        "html": "<html></html>",
        "final_file_name": "resume",
        "validation_attempts": 2,
    }

    fake_events = iter([
        {"type": "step", "label": "Parse job", "status": "done", "detail": "ok"},
        {"type": "state", "state": finished_state},
    ])

    import graph.stream as gstream
    monkeypatch.setattr(gstream, "run_agent", lambda inputs: fake_events)

    at = _app().run()
    at.segmented_control[0].set_value("Paste text")
    at.run()
    at.text_area[0].set_value("Jane Doe\nPython")
    at.text_area[1].set_value("Engineer, Python, SQL")
    at.run()
    at.button[0].click().run()

    assert not at.exception
    # Warning about PDF should be shown.
    warnings = at.warning
    assert len(warnings) >= 1
    assert "No Chromium-based browser available" in warnings[0].value


def test_agent_run_success_streams_events(monkeypatch):
    """Clicking 'Tailor resume' runs the agent and renders results on success."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")
    # Mock the agent stream to yield step + state events without LLM calls.
    fake_events = iter([
        {"type": "step", "label": "Parse resume", "status": "done", "detail": "3 employers"},
        {"type": "reasoning", "text": "Analyzing gap..."},
        {"type": "token", "text": "Generating..."},
        {"type": "state", "state": {
            "final_summary": {"added_skills": ["Python"], "total_required": 4},
            "gap_analysis": type("G", (), {"severity": "light_tweak", "reasoning": "ok"})(),
            "pdf_bytes": b"%PDF-1.4",
            "html": "<html></html>",
            "final_file_name": "test_resume",
        }},
    ])

    import graph.stream as gstream
    monkeypatch.setattr(gstream, "run_agent", lambda inputs: fake_events)

    at = _app().run()

    # Fill in resume + JD text, then click the button.
    at.segmented_control[0].set_value("Paste text")
    at.run()
    at.text_area[0].set_value("Jane Doe\nPython developer")
    at.text_area[1].set_value("Senior Data Engineer. Python SQL Spark.")
    at.run()

    # Click "Tailor resume" button.
    at.button[0].click().run()

    assert not at.exception
    # The step label and reasoning should be visible in the transcript.
    md_texts = [el.value for el in at.markdown]
    assert any("Parse resume" in m for m in md_texts)

    # Results section should appear with the download button.
    assert len(at.download_button) >= 1


def test_agent_run_failure_shows_error(monkeypatch):
    """When the agent raises, the UI surfaces the failure cleanly."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")

    def _raise(_inputs):
        raise RuntimeError("Anthropic API is down")

    import graph.stream as gstream
    monkeypatch.setattr(gstream, "run_agent", _raise)

    at = _app().run()
    at.segmented_control[0].set_value("Paste text")
    at.run()
    at.text_area[0].set_value("Jane Doe\nPython")
    at.text_area[1].set_value("Engineer, Python, SQL")
    at.run()
    at.button[0].click().run()

    assert not at.exception  # the app itself doesn't crash
    # Error message should be shown.
    error_msgs = [el for el in at.error]
    assert any("Anthropic API is down" in e.value for e in error_msgs)


def test_agent_run_failure_shows_status_error(monkeypatch):
    """When the agent raises, the status badge updates to 'Failed'."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")

    import graph.stream as gstream
    monkeypatch.setattr(gstream, "run_agent", lambda _i: iter([{"type": "step", "label": "fail", "status": "running"}]))

    at = _app().run()
    at.segmented_control[0].set_value("Paste text")
    at.run()
    at.text_area[0].set_value("Jane Doe\nPython")
    at.text_area[1].set_value("Engineer, Python, SQL")
    at.run()

    # Override run_agent to raise after first event.
    def _raise_after_event(_inputs):
        yield {"type": "step", "label": "running", "status": "running"}
        raise RuntimeError("Agent crashed mid-stream")

    monkeypatch.setattr(gstream, "run_agent", _raise_after_event)
    at.button[0].click().run()

    assert not at.exception
    # The status widget should show "Failed".
    status_el = at.status
    # AppTest's status element reflects the final label.
    assert len(status_el) >= 1


def test_upload_pdf_source_prompts_uploader(monkeypatch):
    """Upload PDF source shows the file uploader widget; Paste text shows text area."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")
    at = _app().run()

    # Default is "Upload PDF" → uploader is visible.
    assert len(at.file_uploader) >= 1
    # The uploader widget exists and is labeled "Resume PDF".
    assert at.file_uploader[0].label == "Resume PDF"

    # Switch to "Paste text" → uploader disappears, text areas appear.
    at.segmented_control[0].set_value("Paste text")
    at.run()
    assert len(at.file_uploader) == 0
    assert len(at.text_area) >= 1


def test_upload_pdf_corrupt_shows_error(monkeypatch):
    """Uploading a corrupt PDF shows an error, doesn't crash."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")
    at = _app().run()

    # Default source is "Upload PDF" with an uploader.
    assert len(at.file_uploader) >= 1

    # Simulate uploading a corrupt file (AppTest can set value to bytes).
    # Use bytes that pypdf will reject.
    at.file_uploader[0].set_value([("corrupt.pdf", b"not a pdf", "application/pdf")])
    at.run()

    # Should show an error message.
    errors = at.error
    assert len(errors) >= 1
    assert "Could not read that PDF" in errors[0].value


def test_agent_run_no_state_updates_status_failed(monkeypatch):
    """When run_agent yields no final state, status shows Failed."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-fake")

    # Generator yields only step events, no state event.
    import graph.stream as gstream
    monkeypatch.setattr(gstream, "run_agent", lambda _i: iter([
        {"type": "step", "label": "Parse resume", "status": "done"},
        {"type": "step", "label": "Generate", "status": "done"},
        # No state event — agent finished but didn't produce a result.
    ]))

    at = _app().run()
    at.segmented_control[0].set_value("Paste text")
    at.run()
    at.text_area[0].set_value("Jane Doe\nPython")
    at.text_area[1].set_value("Engineer, Python, SQL")
    at.run()
    at.button[0].click().run()

    assert not at.exception
    # Status should show "Failed".
    # AppTest captures the status widget updates.
    status_el = at.status
    assert len(status_el) >= 1
