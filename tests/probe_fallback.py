"""LIVE probe — prove the text-JSON fallback path against the real gateway.

Two checks:
  1. Directly call `_text_json_call` (the exact function invoked when the
     gateway rejects the tool-use schema) and confirm it returns a valid
     ResumeContent from the live free-bundle route.
  2. Run a real `structured_call` with logging on `_text_json_call`, so if the
     gateway happens to reject the schema on this request we SEE the fallback
     fire; otherwise we see the tool-use path succeed.

Run:  python tests/probe_fallback.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from schemas import ResumeContent
from tools import llm

# --- check 1: direct fallback call -------------------------------------------------
t0 = time.time()
client = llm._client()
r = llm._text_json_call(
    client,
    config.MODEL_GENERATE,
    system="You emit a ResumeContent JSON object. Never invent facts.",
    user=(
        "Emit a minimal ResumeContent for Jane Doe: contact, a short summary, "
        "one employer 'Acme Corp' with two bullets, one skill category."
    ),
    output_model=ResumeContent,
    max_tokens=config.GATEWAY_MAX_TOKENS,
    on_token=None,
)
print(
    f"[1] text-JSON fallback OK in {time.time() - t0:.1f}s — "
    f"employers={len(r.employers)} summary_len={len(r.summary)}"
)

# --- check 2: real structured_call with fallback logging ----------------------------
_orig = llm._text_json_call
_orig_tool_use = llm._tool_use_call


def _logged_text_json(*a, **k):
    print("[2] *** text-JSON FALLBACK FIRED (gateway rejected the schema) ***")
    return _orig(*a, **k)


llm._text_json_call = _logged_text_json
t1 = time.time()
try:
    r2 = llm.structured_call(
        model=config.MODEL_GENERATE,
        system="You emit a ResumeContent JSON object. Never invent facts.",
        user=(
            "Tailor this minimal resume for Jane Doe to a Senior Data Engineer "
            "role. Require: Python, Spark, SQL, Kafka, Kubernetes, Airflow. "
            "One employer Acme Corp with client blocks for Northwind Bank and "
            "SkyLine Air, plus a skills section."
        ),
        output_model=ResumeContent,
        max_tokens=config.GATEWAY_MAX_TOKENS,
    )
    print(f"[2] structured_call OK in {time.time() - t1:.1f}s — employers={len(r2.employers)}")
finally:
    llm._text_json_call = _orig
