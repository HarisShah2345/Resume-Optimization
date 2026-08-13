"""Streaming bridge — run the compiled agent and yield live events.

Consumes LangGraph's `stream_mode=["custom", "values"]`: `custom` delivers the
events each node writes through its injected `writer`, and `values` delivers
the merged state after every node. We yield, in order:

  {"type": "step", "label": ..., "status": "running"|"done", "detail": ...}
  {"type": "reasoning", "text": ...}          # gap-analysis reasoning
  {"type": "token", "text": ...}              # raw LLM output tokens
  {"type": "state", "state": <full state>}    # after every node; the LAST one
                                               # carries html/pdf_bytes/final_*

The UI keeps the last `state` event and discards the earlier snapshots.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from graph.agent import build_agent


def build_input(
    resume_text: str,
    jd_text: str,
    *,
    user_emphasis: str | None = None,
    target_title: str | None = None,
) -> dict[str, Any]:
    """Validate + package the user inputs into a graph input dict."""
    if not (resume_text or "").strip() or not (jd_text or "").strip():
        raise ValueError("resume_text and jd_text are required.")
    return {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "user_emphasis": user_emphasis,
        "target_title": target_title,
    }


def run_agent(inputs: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Run the agent once, yielding live events as they happen."""
    app = build_agent()
    for mode, payload in app.stream(inputs, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield payload
        else:  # "values" — full merged state after a node completed
            yield {"type": "state", "state": payload}
