# Resume Optimizer

Production-ready LangGraph agent that tailors a resume to a job description,
with a Streamlit UI. Rebuild of the original n8n "Resume Optimizer" pipeline,
with every hard-won anti-hallucination fix enforced by deterministic code
guardrails rather than prompt instructions alone.

## Model split

| Stage                    | Model            | Notes                                             |
| ------------------------ | ---------------- | ------------------------------------------------- |
| Parse resume structure   | `claude-haiku-4-5` | Cheap structured extraction                      |
| Parse job description    | `claude-haiku-4-5` | Cheap structured extraction                      |
| Generate aligned content | `claude-sonnet-5`  | Main writing stage, matches n8n cost-split       |
| Validate / repair        | `claude-haiku-4-5` | Second cheaper pass, emits corrected JSON        |
| HTML / PDF render        | *(none)*          | Fully deterministic Python / Playwright          |

## Setup

```bash
# 1. Create venv and install dependencies
uv venv --python 3.13
uv pip install -r requirements.txt

# 2. Install the Playwright Chromium browser (one-time)
uv run playwright install chromium

# 3. Configure your API key
cp .env.example .env        # then fill in ANTHROPIC_API_KEY

# 4. Run the UI
uv run streamlit run app/app.py
```

## Tests

```bash
uv run pytest -q
```

The deterministic guardrails (client-detection, keyword cap, alignment gap,
renderer) have full tests that require **no API keys** — they run on canned
fixtures. The LLM-touching tools are thin wrappers over the deterministic core.

## Layout

```
tools/parse_resume.py     # resume_structure extraction + 2-bullet client guardrail
tools/parse_job.py        # jd_data extraction (min counts enforced in code)
tools/alignment_gap.py    # deterministic severity + missing-content analysis
tools/generate_content.py # Sonnet content generation + post-gen code checks
tools/validate_repair.py  # Haiku repair pass (corrected JSON, not a report)
rendering/html_renderer.py# deterministic HTML (registry-ordered, CSS-only bullets)
rendering/pdf_renderer.py # Playwright headless Chromium → Letter PDF
graph/                    # LangGraph agent (state, nodes, edges, streaming)
app/app.py                # Streamlit UI
```
