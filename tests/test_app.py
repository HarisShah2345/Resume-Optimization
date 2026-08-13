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
