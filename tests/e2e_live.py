"""LIVE end-to-end check — requires a real ANTHROPIC_API_KEY.

Runs the full agent against a canned resume + job description and asserts the
production guardrails (the n8n failure modes) on the FINAL output:

  1. a fabricated client never survives to the output,
  2. every JD required skill is present at least once (bug #5 coverage),
  3. no keyword exceeds the code-counted cap (bug #4),
  4. intentionally-added skills (missing from the source) survive validation,
  5. seniority depth floors are met per employer (bug #6),
  6. HTML has an explicit utf-8 charset and NO literal bullet char (ATS),
  7. the PDF is Letter-sized and ATS-clean.

This costs real tokens. Run:

    python tests/e2e_live.py

Exits non-zero if any check fails.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# The failure report uses ✗ and • which cp1252 consoles cannot encode — on a
# Windows codepage that UnicodeEncodeError would crash BEFORE the failing check
# is printed, hiding the very information this script exists to surface. Make
# stdout/stderr encoding-safe so failures are always reported.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Allow running directly (python tests/e2e_live.py) from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from graph.stream import build_input, run_agent
from pypdf import PdfReader
from schemas import ResumeContent
from tools.generate_content import (
    effective_bullet_count,
    seniority_bullet_floors,
)
from tools.guardrails import collect_content_text, count_keyword_mentions

RESUME = """\
Jane Doe
(555) 123-4567
jane.doe@email.com
linkedin.com/in/janedoe

PROFESSIONAL SUMMARY
Data engineer with 7 years designing large-scale data pipelines that cut costs
and improve reliability for high-volume analytics platforms.

PROFESSIONAL EXPERIENCE

Acme Corp | San Francisco, CA | 2019 - Present
Senior Data Engineer
- For the Northwind Bank account, built real-time fraud detection pipelines in Python and Spark.
- For the Northwind Bank account, cut transaction-processing latency 40 percent.
- For the SkyLine Air account, modernized boarding analytics and departure dashboards.
- For the SkyLine Air account, reduced flight-schedule processing time 25 percent.
- Mentored four junior engineers and partnered with analytics stakeholders.

Globex Inc | Austin, TX | 2016 - 2019
Data Engineer
- Built a Python scheduling framework running 200 nightly jobs.
- Standardized SQL reporting used by finance.
- Led a partner engagement for Fabricated MegaCorp on warehouse consolidation.
- Collaborated with marketing analysts on customer lifetime value models.

SKILLS
Python, Spark, SQL, Kafka, AWS Redshift, Terraform
"""

JD_TEXT = """\
Senior Data Engineer

We are hiring a Senior Data Engineer to design, build, and operate reliable,
high-volume data platforms. You will partner with data scientists and
analytics stakeholders to deliver streaming and batch pipelines.

Required skills:
- Python
- Spark
- SQL
- Kafka
- Kubernetes
- Airflow

Key responsibilities:
- Build and maintain ETL and streaming pipelines
- Ensure data quality and reliability across the platform
- Optimize query and storage performance to lower costs
- Collaborate with stakeholders and data scientists
- Operate cloud infrastructure with Kubernetes
- Automate workflow orchestration with Airflow
- Mentor and review the work of junior engineers
- Design for security and compliance

Preferred: AWS, Terraform, Redshift.

Business outcomes: faster delivery, lower costs, higher reliability.
"""


def run_checks(state: dict) -> tuple[list[str], int]:
    """Assert every production guardrail on the final content + rendered output."""
    failures: list[str] = []
    content: ResumeContent = state.get("repaired_content") or state.get("generated_content")
    jd = state["jd_data"]
    gap = state["gap_analysis"]
    html_doc = state["html"]
    pdf_bytes = state["pdf_bytes"]

    text = collect_content_text(content)

    # 1. Fabricated client must never survive as a client block (bug #1).
    # The guardrail's contract is structural: a client only survives if it is
    # grounded AND the parse confirmed >= MIN_BULLETS_PER_CLIENT attributed
    # sentences, otherwise the employer is flattened. The fixture resume
    # deliberately mentions "Fabricated MegaCorp" ONCE in the source — a faithful
    # passing mention surviving as a plain employer bullet is legitimate, but a
    # client BLOCK promoting it is the fabrication this check must catch.
    fabricated = sorted(
        {
            c.name
            for emp in content.employers
            for c in (emp.clients or [])
            if "fabricated megacorp" in c.name.lower()
        }
    )
    if fabricated:
        failures.append(
            f"fabricated client '{', '.join(fabricated)}' survived as a client block"
        )

    # 1b. The two LEGITIMATE clients (Northwind Bank, SkyLine Air — each with
    # >= MIN_BULLETS_PER_CLIENT individually-attributed sentences in the
    # fixture resume) must survive as client blocks. This is the other half
    # of bug #1's contract that had NO live coverage until a real bug shipped
    # undetected: `is_grounded`'s exact-space match flattened every real
    # client on real uploaded PDFs, because `pypdf.extract_text()` routinely
    # turns the space inside a multi-word client name into a newline where it
    # wrapped across a visual line. Catching only "fabricated never survives"
    # missed "real ones get wrongly killed too" — assert both directions.
    surviving = {
        c.name.lower()
        for emp in content.employers
        for c in (emp.clients or [])
    }
    for expected in ("northwind bank", "skyline air"):
        if expected not in surviving:
            failures.append(f"legitimate client '{expected}' was flattened/dropped")

    # 2. JD coverage: every required skill present >= 1.
    # 3. Keyword cap: no required skill > MAX_KEYWORD_FREQ.
    for s in jd.required_skills:
        n = count_keyword_mentions(text, s)
        if n == 0:
            failures.append(f"required skill '{s}' missing from the output")
        elif n > config.MAX_KEYWORD_FREQ:
            failures.append(f"keyword '{s}' appears {n}x (> {config.MAX_KEYWORD_FREQ})")

    # 4. Intentionally-added content must survive validation (bug #5).
    for s in gap.missing_skills:
        if count_keyword_mentions(text, s) == 0:
            failures.append(f"intentionally-added skill '{s}' was stripped by validation")

    # 5. Seniority depth floors per employer (bug #6).
    floors = seniority_bullet_floors(gap.depth_target, len(content.employers))
    for i, emp in enumerate(content.employers):
        n = effective_bullet_count(emp)
        floor = floors[i] if i < len(floors) else floors[-1]
        if n < floor:
            failures.append(f"'{emp.name}' has {n} bullets (floor {floor} for '{gap.depth_target}')")

    # 6. HTML: explicit charset + no literal bullet char (ATS cleanliness).
    if "•" in html_doc:
        failures.append("literal '•' present in the HTML text")
    if 'charset="utf-8"' not in html_doc.lower():
        failures.append("HTML missing utf-8 charset")

    # 7. PDF: Letter size + no literal bullet char.
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pdf_text = "\n".join((p.extract_text() or "") for p in reader.pages)
    if "•" in pdf_text:
        failures.append("literal '•' present in the PDF text")
    sizes = {(float(p.mediabox.width), float(p.mediabox.height)) for p in reader.pages}
    for w, h in sizes:
        if (w, h) != (612.0, 792.0):
            failures.append(f"non-Letter page size: {w}x{h}")
    return failures, len(reader.pages)


def main() -> int:
    inputs = build_input(RESUME, JD_TEXT, target_title="Senior Data Engineer")
    state = None
    print("Running the agent live (this calls the Anthropic API — takes a minute)...\n")
    for event in run_agent(inputs):
        etype = event["type"]
        if etype == "step":
            line = f"  [step] {event.get('label')} · {event.get('status')}"
            if event.get("detail"):
                line += f" · {event['detail']}"
            print(line)
        elif etype == "reasoning":
            print(f"  [reasoning] {event['text']}")
        elif etype == "state":
            state = event["state"]

    if state is None:
        print("FAILED: no final state produced")
        return 1

    gap = state["gap_analysis"]
    summary = state.get("final_summary") or {}
    print(
        f"\nseverity={gap.severity} · validation passes={state['validation_attempts']} "
        f"· skills now present={len(summary.get('added_skills', []))}/{summary.get('total_required', 0)}"
    )

    failures, pages = run_checks(state)
    if failures:
        print("FAILED:")
        for f in failures:
            print("  ✗", f)
        return 1

    print(f"ALL CHECKS PASSED — {pages} Letter page(s), {len(state['pdf_bytes']):,} PDF bytes.")
    print("Added to resume (was missing from the source):", ", ".join(gap.missing_skills))
    return 0


if __name__ == "__main__":
    sys.exit(main())
