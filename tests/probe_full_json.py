"""LIVE probe — full-prompt text-JSON generation on the gateway.

Tests the hypothesis that the SCHEMA-LESS text-JSON path avoids the pathological
auto-think that hangs the tool-use path on a large generation. Uses the exact
generation system prompt and user-context builder as the real agent (the same
inputs that hung 30+ min at 131k on tool-use).

Bounded externally by the caller's `timeout` AND by llm._call_with_retry's own
wall-clock deadline.

Run:  timeout 420 python tests/probe_full_json.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from schemas import (
    ClientProbe,
    EmployerProbe,
    GapAnalysis,
    JobData,
    ResumeContent,
    ResumeStructure,
)
from tools import llm
from tools.generate_content import GENERATION_SYSTEM, build_generation_user

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

JD = JobData(
    role_title="Senior Data Engineer",
    summary_of_role=(
        "Design, build, and operate reliable high-volume data platforms with "
        "streaming and batch pipelines."
    ),
    required_skills=["Python", "Spark", "SQL", "Kafka", "Kubernetes", "Airflow"],
    key_responsibilities=[
        "Build and maintain ETL and streaming pipelines",
        "Ensure data quality and reliability across the platform",
        "Optimize query and storage performance to lower costs",
        "Collaborate with stakeholders and data scientists",
        "Operate cloud infrastructure with Kubernetes",
        "Automate workflow orchestration with Airflow",
    ],
    preferred_qualifications=["AWS", "Terraform", "Redshift"],
    key_phrases=[
        "reliable high-volume data platforms",
        "streaming and batch pipelines",
        "data quality",
        "lower costs",
        "cloud infrastructure",
    ],
    stakeholders=["data scientists", "analytics stakeholders"],
    outcomes=["faster delivery", "lower costs", "higher reliability"],
)

STRUCTURE = ResumeStructure(
    sections_present=["PROFESSIONAL SUMMARY", "PROFESSIONAL EXPERIENCE", "SKILLS"],
    employers=[
        EmployerProbe(
            name="Acme Corp",
            has_clients=True,
            clients=[
                ClientProbe(name="Northwind Bank", bullet_count=2),
                ClientProbe(name="SkyLine Air", bullet_count=2),
            ],
            start_date="2019",
            end_date="Present",
        ),
        EmployerProbe(
            name="Globex Inc",
            has_clients=False,
            start_date="2016",
            end_date="2019",
        ),
    ],
    employer_count=2,
    estimated_seniority="senior",
    contact_fields_found=["name", "phone", "email", "linkedin"],
)

GAP = GapAnalysis(
    severity="full_rebuild",
    missing_skills=["Kubernetes", "Airflow"],
    missing_phrases=["reliable high-volume data platforms", "streaming and batch pipelines"],
    missing_outcomes=[],
    depth_target="senior",
    reasoning=(
        "Full rebuild: the resume covers Python/Spark/SQL/Kafka but lacks the "
        "required Kubernetes and Airflow skills, and its summary needs the JD's "
        "key phrases. Senior depth (6-8 bullets per role) required."
    ),
)

t0 = time.time()
print("full-prompt text-JSON generation start...")
client = llm._client()
r = llm._text_json_call(
    client,
    config.MODEL_GENERATE,
    GENERATION_SYSTEM,
    build_generation_user(
        STRUCTURE, JD, GAP, original_resume_text=RESUME, target_title="Senior Data Engineer"
    ),
    ResumeContent,
    config.GATEWAY_START_TOKENS,
    None,
)
dt = time.time() - t0
print(f"FULL-JSON OK in {dt:.1f}s — employers={len(r.employers)} summary_len={len(r.summary)}")
for e in r.employers:
    bullets = sum(len(c.bullets or []) for c in e.clients) if e.has_clients else len(e.bullets or [])
    print(f"  {e.name}: {e.role_title} — {bullets} bullets")
