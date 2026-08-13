"""LIVE probe — does the new structured_call handle the ResumeContent schema on
the gateway?

If the gateway rejects the tool-use schema (400 "Schema is too complex."), the
text-JSON fallback must fire and return a valid ResumeContent. If the gateway
accepts it, tool-use decoding succeeds instead. Either way the call returns.

Run:  python tests/probe_schema.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from schemas import ResumeContent
from tools.llm import structured_call

print(f"gateway={config.is_gateway()} base={config.API_BASE_URL} model={config.MODEL_GENERATE}")
t0 = time.time()
try:
    result = structured_call(
        model=config.MODEL_GENERATE,
        system="You emit a ResumeContent JSON object. Never invent facts.",
        user=(
            "Emit a minimal ResumeContent for Jane Doe: contact, a short summary, "
            "one employer 'Acme Corp' with two bullets, one skill category."
        ),
        output_model=ResumeContent,
        max_tokens=config.GATEWAY_MAX_TOKENS,
    )
    print(
        f"OK in {time.time() - t0:.1f}s — "
        f"employers={len(result.employers)} summary_len={len(result.summary)}"
    )
except Exception as exc:  # noqa: BLE001 — probe should always report
    print(f"FAILED in {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
    sys.exit(1)
