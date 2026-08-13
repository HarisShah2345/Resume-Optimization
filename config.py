"""Central configuration: env loading, model IDs, and guardrail constants.

Every deterministic threshold that encodes a hard-won n8n fix lives here so it
is tunable in one place and referenced by name in tests.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (cwd-independent: this file's parent).
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# LLM access: direct Anthropic API OR a local gateway/router.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
API_BASE_URL: str | None = os.getenv("ANTHROPIC_BASE_URL") or None

_DIRECT_BASE_URLS = ("", "https://api.anthropic.com", "https://api.anthropic.com/")


def is_gateway() -> bool:
    """True when ANTHROPIC_BASE_URL points at a gateway/router.

    Gateways strip Anthropic's `output_format` structured-output feature, so
    `tools/llm.py` switches to tool-use constrained decoding, and the model
    defaults below resolve to routes the gateway actually serves.
    """
    return (API_BASE_URL or "").strip().rstrip("/") not in _DIRECT_BASE_URLS


# ---------------------------------------------------------------------------
# LLM model split (matches the n8n cost-split architecture)
# ---------------------------------------------------------------------------
# Defaults are the bare Anthropic model names on the direct API; on a gateway
# they fall back to `free-bundle` — the route verified to serve structured
# output on this machine. All three are overridable via env:
# MODEL_PARSE / MODEL_GENERATE / MODEL_VALIDATE.
MODEL_PARSE: str = os.getenv("MODEL_PARSE") or (
    "free-bundle" if is_gateway() else "claude-haiku-4-5"
)
MODEL_GENERATE: str = os.getenv("MODEL_GENERATE") or (
    "free-bundle" if is_gateway() else "claude-sonnet-5"
)
MODEL_VALIDATE: str = os.getenv("MODEL_VALIDATE") or (
    "free-bundle" if is_gateway() else "claude-haiku-4-5"
)

# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------
# Combo gateways AUTO-think even when thinking isn't requested, and a full
# resume generation/correction burns a lot of budget on that thinking. The
# budget choice is a trade-off measured live on free-bundle:
#   - GATEWAY_START_TOKENS (64k): enough headroom that think+emit fit in ONE
#     attempt on the happy path (verified completing in minutes), while small
#     enough that a pathological auto-think is capped. Starting at the ceiling
#     (131k) made the model think for 30+ MINUTES without emitting — strictly
#     worse.
#   - GATEWAY_MAX_TOKENS (131k): the hard ceiling the thinking-only recovery in
#     tools/llm.py grows toward as a bounded backstop.
# Every LLM attempt is also bounded by a wall-clock deadline (see
# tools/llm.py._call_with_retry) so a stalled stream can never hang the agent.
# Direct-API native/tool_use calls stay small — real Sonnet doesn't auto-think
# without being asked THERE. But the text-JSON fallback (tools/llm.py
# `_text_json_call`) is a plain, tool-less completion, and Sonnet was verified
# live to auto-think on it anyway even though thinking is never requested —
# same phenomenon as the gateway, just triggered differently. ResumeContent
# always lands in text-JSON on the direct path (its schema is rejected by
# native `output_format`), so starting that fallback at the small generate
# budget reliably burns it entirely on auto-thinking and forces a wasted
# attempt + bumped retry (measured live: 100-160s wasted per generate/repair
# call). Give the text-JSON fallback the same proven-good start budget as the
# gateway so think+emit usually fit in ONE attempt.
GATEWAY_START_TOKENS: int = 64000
GATEWAY_MAX_TOKENS: int = 131072
DIRECT_MAX_TOKENS_GENERATE: int = 16000
DIRECT_MAX_TOKENS_VALIDATE: int = 16000
DIRECT_TEXT_JSON_START_TOKENS: int = 64000
# Fully disabling thinking on the text-JSON fallback made every generate call
# fast (~12-15s vs 100-345s) but measurably worse at obeying repair
# instructions that need real editing judgment (verified live: the keyword-cap
# repair failed to converge in 3/3 runs, vs. passing when thinking ran
# unbounded). `adaptive` thinking has no budget-token control on claude-sonnet-5
# (`thinking.type.enabled`/`budget_tokens` is rejected: "not supported for this
# model. Use thinking.type.adaptive and output_config.effort" — verified
# live), so the bounded knob is `output_config.effort` instead: capped
# reasoning depth rather than none or unbounded.
DIRECT_TEXT_JSON_THINKING_EFFORT: str = "low"

# ---------------------------------------------------------------------------
# Client-detection guardrail (bug #1) — the single hardest-won fix.
# A named client only "exists" if >= MIN_BULLETS_PER_CLIENT sentences are
# individually attributed to it. Collective mentions count as 0 per client.
# ---------------------------------------------------------------------------
MIN_BULLETS_PER_CLIENT: int = 2

# ---------------------------------------------------------------------------
# JD coverage & keyword discipline (bugs #4 and #5)
# ---------------------------------------------------------------------------
MIN_REQUIRED_SKILLS: int = 6
MIN_KEY_RESPONSIBILITIES: int = 6
MIN_KEY_PHRASES: int = 5
MIN_OUTCOMES: int = 3
MAX_KEYWORD_FREQ: int = 3  # a JD keyword may appear at most 3 times total

# ---------------------------------------------------------------------------
# Role-title reframe cap (bug #7) — never overclaim, never exceed 6 words.
# ---------------------------------------------------------------------------
MAX_ROLE_TITLE_WORDS: int = 6

# ---------------------------------------------------------------------------
# Seniority-scaled depth (bug #6). Senior is a MINIMUM, not a ceiling.
# Maps: estimated_seniority -> (summary_lines, min_bullets_per_employer)
# Bullet floor applies per named client too (validated in post-generation).
# ---------------------------------------------------------------------------
DEPTH_TARGETS: dict[str, dict] = {
    "entry": {"summary_lines": 2, "min_bullets": 3, "max_bullets": 5, "pages": 1},
    "mid": {"summary_lines": 3, "min_bullets": 4, "max_bullets": 6, "pages": 2},
    "senior": {"summary_lines": 4, "min_bullets": 6, "max_bullets": 8, "pages": 3},
}

# ---------------------------------------------------------------------------
# Validation repair loop
# ---------------------------------------------------------------------------
MAX_REPAIR_ITERATIONS: int = 2

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
PAGE_FORMAT: str = "Letter"
PDF_MARGIN_TOP_MM: float = 18.0
PDF_MARGIN_BOTTOM_MM: float = 18.0
PDF_MARGIN_LEFT_MM: float = 14.0
PDF_MARGIN_RIGHT_MM: float = 14.0

TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", Path.home() / ".resume-optimizer"))


def require_api_key() -> str:
    """Return the API key or raise with a clear setup message."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill "
            "in your key, or export ANTHROPIC_API_KEY before running."
        )
    return ANTHROPIC_API_KEY
