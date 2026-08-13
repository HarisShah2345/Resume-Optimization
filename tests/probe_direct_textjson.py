"""LIVE diagnostic — reproduce the direct-API text-JSON fallback and dump the
raw model output, so we can see WHY `no valid JSON` was recovered.

Runs the exact request shape `_text_json_call` sends (no tools, no
output_format, no thinking) against the direct API and prints every content
block, the extraction result, and the validation outcome.

Run:  env -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL python tests/probe_direct_textjson.py
"""
from __future__ import annotations

import sys
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

# Minimal but realistic parsed inputs (mirrors the live parse output).
structure = ResumeStructure(
    sections_present=["PROFESSIONAL SUMMARY", "WORK EXPERIENCE", "SKILLS"],
    employers=[
        EmployerProbe(
            name="Acme Corp",
            has_clients=True,
            clients=[
                ClientProbe(name="Northwind Bank", bullet_count=2),
                ClientProbe(name="SkyLine Air", bullet_count=2),
            ],
        ),
        EmployerProbe(name="Beta Industries", has_clients=False),
    ],
    employer_count=2,
    estimated_seniority="mid",
    contact_fields_found=["name", "phone", "email", "linkedin"],
)
jd = JobData(
    role_title="Senior Data Engineer",
    summary_of_role="Own the data platform: pipelines, warehousing, reliability.",
    required_skills=["Python", "Spark", "SQL", "Kafka", "Kubernetes", "Airflow"],
    key_responsibilities=[
        "Build data pipelines",
        "Design warehouses",
        "Automate CI/CD",
        "Tune performance",
        "Migrate workloads",
        "Secure access",
    ],
    key_phrases=["large-scale data", "reliability", "batch and streaming"],
    outcomes=["reduced pipeline latency", "higher data quality"],
)
gap = GapAnalysis(
    severity="full_rebuild",
    missing_skills=["Kubernetes", "Airflow"],
    missing_phrases=["reliability", "large-scale data"],
    depth_target="mid",
    reasoning="Resume covers 4/6 required skills; full rebuild toward mid depth.",
)
original = """Jane Doe
Senior Data Engineer
(555) 123-4567 | jane.doe@example.com | linkedin.com/in/janedoe

PROFESSIONAL SUMMARY
Data engineer with 8 years building pipelines at Acme Corp.

WORK EXPERIENCE
Acme Corp — Data Engineer (2020-2023)
- Built ETL pipelines in Python and SQL.
- Migrated batch jobs to Spark.

  Northwind Bank:
  - Built real-time fraud alerts for Northwind Bank using Kafka.
  - Cut report turnaround 30% for Northwind Bank.
  SkyLine Air:
  - Designed warehouse schema for SkyLine Air.
  - Automated SkyLine Air's nightly load with Airflow.

Beta Industries — Analyst (2018-2020)
- Analyzed sales data in SQL.
- Automated weekly Excel reports.

SKILLS
Python, SQL, Spark, Kafka, Tableau
"""

user = build_generation_user(
    structure,
    jd,
    gap,
    original_resume_text=original,
    target_title="Senior Data Engineer",
)
prompt = (
    user
    + "\n\nRespond with ONLY a single valid JSON object matching "
    "the required schema. No prose, no markdown fences, no commentary — "
    "just the JSON."
)

client = llm._client()
with client.messages.stream(
    model=config.MODEL_GENERATE,
    max_tokens=config.DIRECT_MAX_TOKENS_GENERATE,
    system=llm._system_arg(GENERATION_SYSTEM),
    messages=[{"role": "user", "content": prompt}],
) as stream:
    final = stream.get_final_message()

print("=== stop_reason:", final.stop_reason)
print("=== usage:", final.usage)
for b in final.content:
    print(f"=== block type={b.type}")
    if getattr(b, "type", None) == "text":
        print(b.text)
    elif getattr(b, "type", None) == "thinking":
        print(f"(thinking block, {len(b.thinking)} chars)")

text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
print("\n=== text len:", len(text))
span = llm._extract_json_object(text)
print("=== extract_json_object ->", (span[:300] + "..." if span and len(span) > 300 else span))
if span:
    try:
        obj = llm._decode_nested_json(span)
        r = ResumeContent.model_validate(obj)
        print(
            "=== VALIDATION OK: employers=%d summary_len=%d"
            % (len(r.employers), len(r.summary))
        )
    except Exception as e:  # noqa: BLE001 — diagnostic
        print("=== VALIDATION FAILED:", type(e).__name__)
        print(str(e)[:2000])
