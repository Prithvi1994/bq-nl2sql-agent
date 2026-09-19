from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Literal
from typing_extensions import TypedDict

from .examples import Example
from .llm import create_llm


SYSTEM = """You are a careful SQLite analyst. You translate a business question into ONE read-only SQL query.

Rules:
- Use only the tables and columns listed in the schema. Never invent names.
- Join only the tables the question needs, using the JOIN conditions given in the hints. Do not add joins that are not required to answer the question.
- SQLite dialect: dates are TEXT like '2023-05-07 00:00:00'; use strftime('%Y', col) for years.
- Convert units carefully: 1 minute = 60000 milliseconds.
- Prefer explicit column lists over SELECT * and give aggregates clear aliases.
- Return the SQL inside a ```sql fenced block and nothing else."""


def _fence(sql: str) -> str:
    return f"```sql\n{sql.strip()}\n```"


def generation_messages(question: str, schema_card: str, join_hints: Sequence[str],
                        examples: Sequence[Example]) -> List[Dict[str, str]]:
    parts = [f"Schema:\n{schema_card}"]
    if join_hints:
        parts.append("Join hints:\n" + "\n".join(f"- {h}" for h in join_hints))
    if examples:
        shots = "\n\n".join(f"Question: {ex.question}\n{_fence(ex.sql)}" for ex in examples)
        parts.append(f"Examples of the house style:\n\n{shots}")
    parts.append(f"Question: {question}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def repair_messages(question: str, schema_card: str, join_hints: Sequence[str],
                    previous_sql: str, errors: Sequence[str]) -> List[Dict[str, str]]:
    problems = "\n".join(f"- {e}" for e in errors)
    user = (
        f"Schema:\n{schema_card}\n\n"
        + ("Join hints:\n" + "\n".join(f"- {h}" for h in join_hints) + "\n\n" if join_hints else "")
        + f"Question: {question}\n\n"
        f"Your previous query:\n{_fence(previous_sql)}\n\n"
        f"It failed for these reasons:\n{problems}\n\n"
        "Write a corrected query that fixes every listed problem. Return only the ```sql block."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def judge_messages(question: str, reference_sql: str, reference_preview: str,
                   candidate_sql: str, candidate_preview: str) -> List[Dict[str, str]]:
    user = (
        "Two SQL queries were written for the same question. Decide whether the CANDIDATE answers the "
        "question as correctly as the REFERENCE, judging by meaning and by the result previews. Minor "
        "differences in column names, ordering or formatting do not matter; different entities, filters, "
        "aggregations or row sets do.\n\n"
        f"Question: {question}\n\n"
        f"REFERENCE SQL:\n{reference_sql}\nREFERENCE result preview:\n{reference_preview}\n\n"
        f"CANDIDATE SQL:\n{candidate_sql}\nCANDIDATE result preview:\n{candidate_preview}\n\n"
        "Answer with exactly one word on the first line, YES or NO, then one sentence of reason."
    )
    return [{"role": "system", "content": "You are a strict but fair SQL reviewer."},
            {"role": "user", "content": user}]


_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> Optional[str]:
    """Pull the SQL out of a model reply: fenced block first, else the first SELECT/WITH statement."""
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip().rstrip(";").strip()
    m = re.search(r"(?is)\b(select|with)\b.*", text)
    if m:
        candidate = m.group(0).split(";")[0].strip()
        return candidate or None
    return None


def parse_judge(text: str) -> Optional[bool]:
    first = (text or "").strip().splitlines()[0].strip().upper() if (text or "").strip() else ""
    if first.startswith("YES"):
        return True
    if first.startswith("NO"):
        return False
    return None


# ============================================================================
# REPORT GENERATION PROMPTS
# ============================================================================

REPORT_SYSTEM = """You are a senior data analyst writing a professional report from structured data.

Rules:
- Base every claim on the provided data. No external knowledge.
- Be precise with numbers: cite exact values from the data.
- Structure with clear sections and headers.
- Note limitations or data gaps explicitly.
- Return ONLY the report in markdown format with ## section headers."""


def report_generation_messages(
    question: str,
    sql: str,
    data_preview: str,
    style: str,
    row_count: int,
) -> List[Dict[str, str]]:
    style_guidance = {
        "executive_summary": "Write a concise executive summary with key findings, metrics, and recommendations. 2-3 sections max.",
        "detailed_analysis": "Write a comprehensive analysis with methodology, detailed findings per segment, and appendix. 4-6 sections.",
        "technical_brief": "Write a technical brief with query logic, data quality notes, and reproducibility details. 3-4 sections.",
        "slide_deck": "Write slide-ready bullets: title, key metric, insight per slide. 5-8 slides.",
    }
    
    parts = [
        f"Question: {question}",
        f"SQL Query:\n```sql\n{sql}\n```",
        f"Data ({row_count} rows):\n{data_preview}",
        f"Style: {style_guidance.get(style, style_guidance['executive_summary'])}",
    ]
    
    return [
        {"role": "system", "content": REPORT_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def report_review_messages(
    question: str,
    current_report: str,
    feedback: str,
    style: str,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": REPORT_SYSTEM + "\n\nRevise the report based on reviewer feedback. Preserve all grounded claims."},
        {"role": "user", "content": f"Original Question: {question}\n\nCurrent Report:\n{current_report}\n\nReviewer Feedback:\n{feedback}\n\nStyle: {style}\n\nProvide the revised report."},
    ]


def _parse_report_sections(report_text: str) -> List[Dict]:
    """Parse report into sections."""
    import re
    sections = []
    
    # Try to find markdown headers
    parts = re.split(r'\n##\s+', report_text)
    for i, part in enumerate(parts):
        if i == 0 and not part.strip().startswith("#"):
            # First part might be intro
            if part.strip():
                sections.append({"title": "Introduction", "content": part.strip()})
        else:
            lines = part.strip().split("\n")
            title = lines[0] if lines else f"Section {i}"
            content = "\n".join(lines[1:]) if len(lines) > 1 else ""
            sections.append({"title": title, "content": content})
    
    return sections


def _extract_summary(report_text: str) -> str:
    """Extract or generate summary."""
    lines = report_text.strip().split("\n")
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped and not stripped.startswith("#"):
            return stripped[:500]
    return "Report generated from data analysis."


# ============================================================================
# REPORT GRADER / EVALUATOR
# ============================================================================

@dataclass
class GraderResult:
    """Result of report grading."""
    decision: Literal["approve", "revise", "reject"]
    scores: Dict[str, float]  # groundedness, faithfulness, factual_precision, completeness, style
    feedback: str  # Specific actionable feedback
    overall: float  # Weighted average


def grade_report(
    question: str,
    report: str,
    sql: str,
    columns: List[str],
    rows: List[tuple],
    style: str,
    model: str = "gpt-4o-mini",
) -> GraderResult:
    """
    Grade a generated report across multiple dimensions.
    
    Evaluates:
    - groundedness: Every claim supported by source data
    - faithfulness: No contradictions with source data
    - factual_precision: Numbers match source exactly
    - completeness: Covers key insights (totals, trends, outliers, methodology)
    - style_adherence: Matches requested style/format
    
    Returns GraderResult with decision, scores, and actionable feedback.
    """
    # For mock testing, return perfect scores
    if "mock" in str(model).lower():
        return GraderResult(
            decision="approve",
            scores={
                "groundedness": 1.0,
                "faithfulness": 1.0,
                "factual_precision": 1.0,
                "completeness": 1.0,
                "style_adherence": 1.0,
                "overall": 1.0,
            },
            feedback="All checks passed (mock mode)",
            overall=1.0,
        )
    
    # Real grading using LLM judge
    llm = create_llm(model)
    
    # Format source data for judge
    data_preview = _format_data_for_llm(columns, rows, max_rows=20)
    
    # Build grading prompt
    grading_prompt = GRADER_SYSTEM + "\n\n" + grader_messages(
        question=question,
        report=report,
        sql=sql,
        data_preview=data_preview,
        style=style,
    )
    
    reply = llm.complete([{"role": "system", "content": GRADER_SYSTEM}, 
                           {"role": "user", "content": grading_prompt}])
    
    return _parse_grader_reply(reply)


def _parse_grader_reply(reply: str) -> GraderResult:
    """Parse grader LLM reply into structured result."""
    import json, re
    
    # Try to extract JSON from reply
    json_match = re.search(r'```json\s*(.*?)\s*```', reply, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return GraderResult(
                decision=data.get("decision", "revise"),
                scores=data.get("scores", {}),
                feedback=data.get("feedback", ""),
                overall=data.get("overall", 0.0),
            )
        except json.JSONDecodeError:
            pass
    
    # Fallback: parse text format
    lines = reply.strip().split("\n")
    decision = "revise"
    scores = {}
    feedback = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith("DECISION:"):
            decision = line.split(":", 1)[1].strip().lower()
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
        elif ":" in line and any(k in line.lower() for k in ["groundedness", "faithfulness", "factual", "completeness", "style", "overall"]):
            key, val = line.split(":", 1)
            try:
                scores[key.strip().lower().replace(" ", "_")] = float(val.strip())
            except ValueError:
                pass
    
    if not scores.get("overall"):
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 0.0
    
    return GraderResult(
        decision=decision if decision in ("approve", "revise", "reject") else "revise",
        scores=scores,
        feedback=feedback or "See grader output",
        overall=scores.get("overall", 0.0),
    )


# ============================================================================
# GRADER PROMPTS
# ============================================================================

GRADER_SYSTEM = """You are an expert report quality evaluator. You evaluate data analysis reports for quality across five dimensions:

1. GROUNDEDNESS (0-1): Every claim in the report is supported by the provided source data. No external knowledge or hallucination.
2. FAITHFULNESS (0-1): The report does not contradict the source data. Numbers, trends, and statements align with the actual data.
3. FACTUAL_PRECISION (0-1): Specific numbers, percentages, and calculations in the report exactly match the source data.
4. COMPLETENESS (0-1): The report covers all key insights: totals, trends, outliers, segment breakdowns, methodology.
5. STYLE_ADHERENCE (0-1): The report matches the requested style (executive summary, detailed analysis, technical brief, slide deck).

Return a JSON object with:
{
  "decision": "approve" | "revise" | "reject",
  "scores": {
    "groundedness": 0.0-1.0,
    "faithfulness": 0.0-1.0,
    "factual_precision": 0.0-1.0,
    "completeness": 0.0-1.0,
    "style_adherence": 0.0-1.0,
    "overall": 0.0-1.0
  },
  "feedback": "Specific, actionable feedback for revision. Cite exact issues.",
  "overall": 0.0-1.0
}

Decision thresholds:
- APPROVE: overall >= 0.85 AND all dimensions >= 0.7
- REVISE: overall >= 0.6 OR any dimension < 0.7
- REJECT: overall < 0.6 OR factual_precision < 0.5 OR groundedness < 0.5

Be strict but constructive. Cite exact numbers and claims from the report that fail."""


def grader_messages(
    question: str,
    report: str,
    sql: str,
    data_preview: str,
    style: str,
) -> str:
    return f"""Evaluate this report for the question: "{question}"

SQL QUERY:
```sql
{sql}
```

SOURCE DATA:
{data_preview}

REPORT TO EVALUATE:
{report}

REQUESTED STYLE: {style}

Evaluate the report across all five dimensions. Be strict and specific. Return JSON only."""


def _format_data_for_llm(columns: List[str], rows: List[tuple], max_rows: int = 100) -> str:
    """Format data for LLM consumption."""
    if not rows:
        return "(no data)"
    
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in r) for r in rows[:max_rows])
    more = f"\n... ({len(rows)} rows total)" if len(rows) > max_rows else ""
    return f"{header}\n{body}{more}"


# Export for use in graph
__all__ = [
    "GraderResult",
    "grade_report",
    "grader_messages",
    "GRADER_SYSTEM",
    "report_generation_messages",
    "report_review_messages",
    "_parse_report_sections",
    "_extract_summary",
]