"""
Evaluation harness: run a golden set through the agent and score it three ways.

  execution success   the generated SQL ran without error
  result match        its rows equal the reference query's rows (order-insensitive, values normalised)
  judged equivalent   optional: an LLM judge says the candidate answers the question as well as the
                      reference, used only where result match failed (different but valid formulations)

The primary metric is result match because it is deterministic and reproducible.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .agent import AskResult, NL2SQLAgent
from .executor import run_query
from .llm import LLM
from .prompts import judge_messages, parse_judge


@dataclass
class GoldenCase:
    id: str
    question: str
    sql: str
    tags: List[str] = field(default_factory=list)


def load_golden(path: str) -> List[GoldenCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenCase(id=str(d["id"]), question=d["question"], sql=d["sql"], tags=list(d.get("tags", []))) for d in data]


# ---------------------------------------------------------------------------
def _norm_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip().casefold()
        try:
            return round(float(s), 2)
        except ValueError:
            return s
    return v


def normalise_rows(rows: Sequence[Sequence[Any]]) -> Counter:
    """Multiset of rows with normalised values, so ordering and formatting do not matter."""
    return Counter(tuple(_norm_value(v) for v in r) for r in rows)


def rows_match(candidate: Sequence[Sequence[Any]], reference: Sequence[Sequence[Any]]) -> bool:
    return normalise_rows(candidate) == normalise_rows(reference)


def preview(columns: Sequence[str], rows: Sequence[Sequence[Any]], n: int = 5) -> str:
    head = " | ".join(columns) if columns else "(no columns)"
    body = "\n".join(" | ".join(str(v) for v in r) for r in list(rows)[:n])
    more = f"\n... ({len(rows)} rows total)" if len(rows) > n else ""
    return f"{head}\n{body}{more}"


# ---------------------------------------------------------------------------
@dataclass
class CaseOutcome:
    id: str
    question: str
    status: str
    exec_ok: bool
    match: bool
    judged: Optional[bool]
    repairs: int
    confidence: float
    total_ms: int
    candidate_sql: Optional[str]
    reference_sql: str
    candidate_rows: int
    reference_rows: int
    error: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    llm: str
    cases: List[CaseOutcome]
    started: str
    elapsed_s: float

    @property
    def n(self) -> int:
        return len(self.cases)

    def rate(self, attr: str) -> float:
        return sum(1 for c in self.cases if getattr(c, attr)) / self.n if self.n else 0.0

    def summary(self) -> Dict[str, Any]:
        judged_true = sum(1 for c in self.cases if c.judged)
        accepted = sum(1 for c in self.cases if c.match or c.judged)
        return {
            "llm": self.llm,
            "cases": self.n,
            "execution_success": self.rate("exec_ok"),
            "result_match": self.rate("match"),
            "judge_rescued": judged_true,
            "accepted": accepted / self.n if self.n else 0.0,
            "repairs_histogram": dict(sorted(Counter(c.repairs for c in self.cases).items())),
            "median_ms": int(statistics.median(c.total_ms for c in self.cases)) if self.n else 0,
            "elapsed_s": round(self.elapsed_s, 1),
        }

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# NL2SQL evaluation: {self.llm}",
            "",
            f"Run started {self.started}, {s['cases']} cases, {s['elapsed_s']} s wall clock.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Execution success | {s['execution_success']:.0%} |",
            f"| Result match (deterministic) | {s['result_match']:.0%} |",
            f"| Judge-rescued cases | {s['judge_rescued']} |",
            f"| Accepted (match or judged equivalent) | {s['accepted']:.0%} |",
            f"| Repairs used (count by n) | {s['repairs_histogram']} |",
            f"| Median latency per question | {s['median_ms']} ms |",
            "",
            "| # | Question | Status | Exec | Match | Judge | Repairs | ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in self.cases:
            judge = "" if c.judged is None else ("yes" if c.judged else "no")
            lines.append(f"| {c.id} | {c.question} | {c.status} | {'✓' if c.exec_ok else '✗'} | "
                         f"{'✓' if c.match else '✗'} | {judge} | {c.repairs} | {c.total_ms} |")
        misses = [c for c in self.cases if not c.match]
        if misses:
            lines += ["", "## Mismatches", ""]
            for c in misses:
                lines += [f"### {c.id}: {c.question}", "", "Reference:", "```sql", c.reference_sql, "```",
                          "Candidate:", "```sql", c.candidate_sql or "(none)", "```"]
                if c.error:
                    lines += [f"Error: `{c.error}`"]
                lines.append("")
        return "\n".join(lines)


def evaluate(agent: NL2SQLAgent, golden: Sequence[GoldenCase], *, judge: Optional[LLM] = None,
             progress=None) -> EvalReport:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    cases: List[CaseOutcome] = []
    iterator = progress(golden) if progress else golden
    for g in iterator:
        ref = run_query(agent.db_path, g.sql, row_limit=agent.row_limit, timeout_s=agent.timeout_s)
        if not ref.ok:
            raise ValueError(f"Golden case {g.id} reference SQL failed: {ref.error}")
        res: AskResult = agent.ask(g.question)
        exec_ok = res.status != "failed"
        match = exec_ok and rows_match(res.rows, ref.rows)
        judged: Optional[bool] = None
        if judge is not None and exec_ok and not match:
            verdict = judge.complete(judge_messages(g.question, g.sql, preview(ref.columns, ref.rows),
                                                    res.sql or "", preview(res.columns, res.rows)))
            judged = parse_judge(verdict)
        last = res.attempts[-1]
        error = "; ".join(last.guard_errors) or last.exec_error
        cases.append(CaseOutcome(id=g.id, question=g.question, status=res.status, exec_ok=exec_ok, match=match,
                                 judged=judged, repairs=res.repairs_used, confidence=res.confidence,
                                 total_ms=res.total_ms, candidate_sql=res.sql, reference_sql=g.sql,
                                 candidate_rows=len(res.rows), reference_rows=len(ref.rows), error=error, tags=g.tags))
    return EvalReport(llm=getattr(agent.llm, "name", "?"), cases=cases, started=started,
                      elapsed_s=time.perf_counter() - t0)


def save_report(report: EvalReport, md_path: str, json_path: Optional[str] = None) -> None:
    Path(md_path).write_text(report.to_markdown(), encoding="utf-8")
    if json_path:
        Path(json_path).write_text(json.dumps({"summary": report.summary(), "cases": [asdict(c) for c in report.cases]},
                                              indent=2, default=str), encoding="utf-8")
