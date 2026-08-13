# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A production-ready LangGraph agent that tailors a resume to a job description, with a Streamlit UI. It is a rebuild of an original n8n "Resume Optimizer" pipeline where every hard-won anti-hallucination fix is enforced by **deterministic code guardrails rather than prompt instructions alone**. The LLM proposes; code verifies and enforces.

## Commands

Run from the project root. All commands use `uv` (project env lives in `.venv/`).

```bash
# Setup
uv venv --python 3.13
uv pip install -r requirements.txt
uv run playwright install chromium      # one-time; PDF rendering needs a browser
cp .env.example .env                    # fill in ANTHROPIC_API_KEY

# Run the UI
uv run streamlit run app/app.py

# Run all tests (deterministic tests need NO API key; LLM tools are mocked)
uv run pytest -q

# Run one test file / one test
uv run pytest -q tests/test_agent.py
uv run pytest -q tests/test_agent.py::test_repair_loop_is_bounded_and_still_renders
```

The `probe_*.py` and `e2e_live.py` files under `tests/` are **live scripts that cost real tokens** (they call the Anthropic API). They are run directly, not via pytest:
`python tests/e2e_live.py`, `python tests/probe_schema.py`, `python tests/probe_fallback.py`, `python tests/probe_full_json.py`, `python tests/probe_direct_textjson.py`.

### Disk space (Windows)

The C drive is full on this machine (0 GB free); D has plenty. The project, `.venv/`, and `.uv-cache/` all live on D, and `.env` pins the two C-drive-bound paths to D:
- `TEMP_DIR` → `D:\n8n Workflows\Resume-Optimizer\.runtime\tmp` (used for `tools/llm.py`'s `llm.log`).
- `PLAYWRIGHT_BROWSERS_PATH` → `D:\n8n Workflows\Resume-Optimizer\.playwright-browsers`.

If you ever need to install Chromium, set `PLAYWRIGHT_BROWSERS_PATH` first so it lands on D (it otherwise defaults to `%LOCALAPPDATA%\ms-playwright` on C): `$env:PLAYWRIGHT_BROWSERS_PATH="D:\n8n Workflows\Resume-Optimizer\.playwright-browsers"; uv run playwright install chromium`. Keep uv on the existing on-D `.uv-cache`; don't let it fall back to `%LOCALAPPDATA%\uv`.

## Architecture

A LangGraph DAG of single-purpose nodes; each wraps one tool and streams live events to the UI.

```
START → parse_resume → parse_job → check_gap → generate → validate
                                                 ↑            │
                                                 └── repair ──┘  (conditional)
                                                      ↓
                                 render → finalize → END
```

### Data flow (follow the state to understand the pipeline)

- **Inputs**: `resume_text`, `jd_text`, optional `user_emphasis` / `target_title`.
- **Parse**: Haiku extracts `ResumeStructure` and `JobData` (structured output). Client-detection filter (`filter_clients_by_min_bullets`) runs in code immediately after.
- **Gap**: `check_alignment_gap` computes severity + missing content **deterministically** (keyword presence against the original resume text) — no LLM judgment call.
- **Generate**: Sonnet writes the tailored resume, streamed with live tokens.
- **Post-checks**: `run_post_generation_checks` runs every deterministic guardrail in code, auto-fixing what's safe and producing `repair_instructions` otherwise.
- **Validate**: a cheaper Haiku pass emits a **corrected** `ResumeContent` in the same schema (never a prose report), told exactly what was `intentionally_added` so it won't strip JD coverage (bug #5). Its output is re-run through the same post-checks; if that made a guardrail-clean generation **worse**, the pre-validation generation is kept instead (regression guard in `validate_node`).
- **Repair loop**: the conditional edge `should_repair` regenerates only when post-checks found fixable violations AND repair budget remains (`MAX_REPAIR_ITERATIONS=2`). Bounded — the candidate always gets a document.
- **Render + finalize**: fully deterministic HTML → PDF (Playwright), then a diff summary + download filename.

### Key files

- `schemas.py` — the Pydantic v2 contracts every stage agrees on. Used for BOTH structured-output validation and deterministic post-processing.
- `config.py` — env loading, model IDs, and **every deterministic threshold** (keyword cap, bullet floors, token budgets). Referenced by name in tests so constants stay tunable in one place.
- `tools/llm.py` — the only place that talks to the LLM. One `structured_call` entry point with three mechanisms (see below).
- `tools/guardrails.py` — pure deterministic checks, no LLM, each tied to a numbered bug fixed from the n8n pipeline (see below). Bugs #1 (client grounding), #4 (keyword cap), and #7 (compound role title) all have deterministic **auto-fixes**, not just detection.
- `tools/generate_content.py` — Sonnet generation + `run_post_generation_checks` (the post-gen code guarantees).
- `tools/validate_repair.py` — the Haiku validation/repair pass.
- `tools/parse_resume.py` / `tools/parse_job.py` / `tools/alignment_gap.py` — parse + deterministic gap stages.
- `rendering/html_renderer.py` — deterministic HTML: explicit UTF-8 charset, sections in `sections_present` order, precise `page-break-inside:avoid` atomicity rules. Bullets/client-name markers are a painted CSS shape (`.bullet-dot`: a textless `<span>` styled as a filled circle, absolutely positioned with an em-based offset) — never a `content:"•"` character, which bakes a literal "•" into the PDF's extractable text layer (verified live) and can trip up ATS parsers. Ported several n8n visual details this rebuild had dropped: a `header-title` job-title subtitle under the candidate's name, an `<hr>` under the header, the 2-row company header (name+dates / role+location), bold (not italic) dates, and right-aligned certification years. `.role-title` is italic only, not bold (later preference, not an n8n-parity item).
- `rendering/pdf_renderer.py` — Playwright headless Chromium → Letter PDF; resolves bundled Chromium → system Edge → Chrome.
- `graph/state.py` / `graph/nodes.py` / `graph/agent.py` / `graph/stream.py` — LangGraph wiring. `step_log` uses an `operator.add` reducer (appends); everything else is last-wins.
- `app/app.py` — Streamlit UI; inserts the project root into `sys.path` so it can be launched from anywhere. Native theming only (`.streamlit/config.toml`, shadcn-style, light+dark) — no custom CSS. Material Symbols icons, bordered-card sections, `st.segmented_control` over `st.radio`, bordered `st.metric` tiles, badge-based severity/skill display.
- `.streamlit/config.toml` — `[theme.light]` / `[theme.dark]` (both required for the mode switcher to appear in the app's settings menu); shared font/radius settings live in the bare `[theme]` table.

## Core design rules

- **Deterministic guardrails over prompt instructions.** The numbered bugs (#1–#7 and #9 — the numbering skips #8) in comments throughout the code are the n8n failure modes this rebuild exists to fix. The most important: **#1 client fabrication** (a named client only survives if >= `MIN_BULLETS_PER_CLIENT` sentences are individually attributed to it, AND the name is grounded as a whole word in the original resume text — otherwise the employer is flattened), **#2 `sections_present` is the sole authority** for employer set and section order (generation + rendering must match it 1:1), **#4 keyword frequency cap** (`MAX_KEYWORD_FREQ=3`, counted across the whole generated text in code), **#7 role-title reframe cap** (`MAX_ROLE_TITLE_WORDS=6`, unique per employer). Before changing any threshold, update the name in `config.py` and its tests.
  - **Bug #1's grounding check is whitespace-tolerant, not exact-space.** `guardrails.is_grounded` matches whitespace between a multi-word name's words as `\s+`, not a literal space. This was a real shipped bug, not theoretical: `resume_text` comes from `pypdf.extract_text()` on an uploaded PDF, which routinely turns the space inside a client name into a newline (or extra spaces) when it wrapped across a visual line or column boundary — and since `enforce_client_grounding` flattens an employer's ENTIRE client list if even ONE client fails grounding, this was wiping out every real client on real uploaded resumes (verified live) while `tests/e2e_live.py` only ever checked that a *fabricated* client doesn't survive, never that *legitimate* ones do — so it shipped undetected. Both directions are now asserted live (see Testing conventions).
  - **Bug #4 has a deterministic auto-fix**, added after live testing showed the LLM repair pass converging on the keyword-cap violation unreliably (3/3 failures with thinking disabled, 1 pass/1 fail at `medium` effort) — it's a precise-counting/editing task the model doesn't reliably nail regardless of `thinking` effort level. `guardrails.enforce_keyword_cap` removes redundant TAG-style mentions — skills-category items, employer/client `tools` lists — never bullet prose, so no sentence is ever mangled; bullets keep the real evidence for a skill, so a duplicate tag is pure keyword-stuffing repetition and safe to drop outright. Only keywords still over the cap after that (living only in bullet prose) become a `repair_instruction` for the LLM. Wired into `generate_content.run_post_generation_checks` the same way bug #1's grounding fix is.
  - **Bug #7 also has a deterministic auto-fix** for a failure mode the word cap alone doesn't catch: a compound, comma-separated role title ("Senior Data Engineer, Cloud ETL") that's still <= `MAX_ROLE_TITLE_WORDS`. `guardrails.enforce_single_role_title` unconditionally collapses it to the primary title before the comma — splits ONLY on comma, never a hyphen, so "Full-Stack Engineer" is untouched. The generation prompt also asks for a single title up front, but the guardrail is what actually guarantees it.
- **Never invent facts.** The generator's ground truth is `original_resume_text`; JD content genuinely absent from the source may be added as ONE new standalone bullet, never stretched onto an existing sentence. Validation is the only stage allowed to see both.
- **Gateway vs direct API.** `config.is_gateway()` switches model defaults (`free-bundle` route vs `claude-haiku-4-5` / `claude-sonnet-5`), the structured-output mechanism, token budgets, and thinking behavior. Gateways auto-think even when not asked, so on a gateway, generation starts at `GATEWAY_START_TOKENS` (64k) and `thinking` is disabled; `tools/llm.py` recovers a thinking-only response exactly once by disabling thinking and bumping toward `GATEWAY_MAX_TOKENS`.
- **`tools/llm.py` has three structured-output mechanisms.** NATIVE (direct API: `output_format` Pydantic), GATEWAY (forced tool-use constrained decoding — the universally-supported technique), and TEXT-JSON last resort (no tools; the model emits JSON as text, recovered by extracting the outermost `{...}` span — fires when a gateway rejects the tool schema with `400 Schema is too complex.`). The direct-API native path shares the same grammar machinery, so a rejected schema falls straight to text-JSON too.
  - **Schema rejection is memoized twice**: in-process (`_native_schema_failures`, keyed by `id(output_model)`) so every repair/validate pass in one run skips the doomed native attempt, AND **persisted to disk** (`_persisted_schema_failures`, keyed by class name, backed by `<TEMP_DIR>/native_schema_failures.json`) so a *fresh* process — a new Streamlit rerun, a new `e2e_live.py` invocation — never re-pays the rejection round-trip either. That round-trip is real and highly variable: verified live between 3 and 26 minutes for the exact same schema (`ResumeContent`), since it's a server-side grammar-compile cost outside our control. Delete the JSON file to force re-discovery (e.g. after a genuinely new schema shape).
  - **`text_json`'s own auto-thinking is bounded, not disabled.** Sonnet auto-thinks on this tool-less path even when `thinking` is never requested in the API call — verified live. Fully disabling it (`{"type": "disabled"}`) made every call ~20x faster (100-345s → 12-15s) but measurably worse at obeying edit-heavy repair instructions (3/3 live runs failed to converge the keyword-cap repair — since fixed with the code-side auto-fix above, not by restoring thinking). `thinking.type.enabled` + `budget_tokens` is rejected outright on `claude-sonnet-5` (`400`: "not supported for this model. Use thinking.type.adaptive and output_config.effort" — verified live), so the actual bounded knob is `output_config.effort` (`config.DIRECT_TEXT_JSON_THINKING_EFFORT`, currently `"low"`) alongside `thinking: {"type": "adaptive"}` — bounded reasoning depth instead of none or unbounded. `config.DIRECT_TEXT_JSON_START_TOKENS` (64000, matching `GATEWAY_START_TOKENS`) is the other half of the fix: starting `text_json` at the small native generate budget (16000) reliably burned the whole thing on auto-thinking before any JSON was emitted, forcing a bumped-retry every time.
  - Transient 5xx/pool errors retry with backoff; every attempt is bounded by a wall-clock deadline so a hung stream can never block the agent.
- **Streaming is first-class.** Every LLM node passes `on_token` through; `graph/stream.py` yields `step` / `reasoning` / `token` / `state` events via `stream_mode=["custom", "values"]`. The UI keeps only the LAST `state` event. A failed attempt's buffered deltas are discarded, never leaked to the UI.

## Testing conventions

- The deterministic guardrails (client detection, keyword cap, alignment gap, renderer, post-checks) have **full test coverage with no API keys** — they run on canned fixtures.
- LLM-touching tools are **thin wrappers** over the deterministic core, so their tests mock the LLM call and assert the plumbing (correct schema/model, guardrail applied to the returned structure).
- `test_llm.py` exercises the REAL `structured_call` code against a fake `client.messages.stream` — the two structured-output mechanisms, thinking-only recovery, schema-too-complex fallback, retry-with-backoff, and wall-clock timeout. Its autouse fixture clears BOTH the in-process (`_native_schema_failures`) and persisted (`_persisted_schema_failures`) rejection caches and redirects `config.TEMP_DIR` to a pytest `tmp_path` — tests must never read or write the real on-disk `native_schema_failures.json`, and the shared `Sample` schema class must not leak rejection state between tests.
- `test_agent.py` mocks all LLM tools + `render_pdf`, runs the real compiled graph, and proves the agent's *shape*: node order, bounded repair loop, `intentionally_added` + `repair_instructions` propagation, live event streaming.
- `test_app.py` uses Streamlit's `AppTest` harness (no browser) for UI smoke tests.
- Tests must be run from the project root (imports are top-level, e.g. `import config`).
- The real correctness proof is `tests/e2e_live.py` — runs the full agent with a real API key and asserts every production guardrail on the final output (fabricated client never survives, **legitimate clients — Northwind Bank, SkyLine Air — DO survive** [added after the whitespace-grounding bug shipped undetected; assert both directions of a flatten/keep guardrail, not just one], all JD skills present, keyword cap, intentionally-added content survives validation, depth floors, ATS-clean HTML/PDF, Letter page size).
- `test_client_guardrail.py::test_grounded_tolerates_pdf_extraction_whitespace_noise` is the regression test for the whitespace-grounding bug — covers newline/double-space noise inside a multi-word name, while still rejecting words genuinely joined with no separator.
