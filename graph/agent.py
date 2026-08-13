"""StateGraph assembly — the agent's control flow.

```
START → parse_resume → parse_job → check_gap → generate → validate
                                                     ↑            │
                                                     └── repair ──┘   (conditional)
                                                          ↓
                                     render → finalize → END
```

The only conditional edge is the **repair loop** (a genuine agent decision, not
a fixed pipeline): after every generate + validate, the deterministic
post-check report decides whether the content is good enough or whether a
fresh generation pass is needed with the specific repair instructions. The
loop is bounded by `config.MAX_REPAIR_ITERATIONS` (default 2), so worst case
the graph runs 3 generation passes and then renders whatever it has — the
candidate always gets a document back.
"""
from __future__ import annotations

import config
from graph.nodes import (
    check_gap_node,
    finalize_node,
    generate_node,
    parse_job_node,
    parse_resume_node,
    render_node,
    validate_node,
)
from graph.state import ResumeAgentState
from langgraph.graph import END, START, StateGraph


def should_repair(state: ResumeAgentState) -> str:
    """Conditional edge after `validate`.

    Regenerate only when the deterministic post-checks found violations that
    carry a concrete repair instruction AND we still have repair budget.
    Otherwise render whatever we have. Bounded: at most MAX_REPAIR_ITERATIONS
    repair regenerations (validation_attempts counts completed validate passes).
    """
    report = state.get("postcheck_report")
    failing = report is not None and not report.passed and bool(report.repair_instructions)
    attempts = state.get("validation_attempts", 0)
    if failing and attempts <= config.MAX_REPAIR_ITERATIONS:
        return "repair"
    return "render"


def build_agent():
    """Compile the LangGraph StateGraph. Caller chooses invoke() or stream()."""
    builder = StateGraph(ResumeAgentState)

    builder.add_node("parse_resume", parse_resume_node)
    builder.add_node("parse_job", parse_job_node)
    builder.add_node("check_gap", check_gap_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("render", render_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "parse_resume")
    builder.add_edge("parse_resume", "parse_job")
    builder.add_edge("parse_job", "check_gap")
    builder.add_edge("check_gap", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        should_repair,
        {"repair": "generate", "render": "render"},
    )
    builder.add_edge("render", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
