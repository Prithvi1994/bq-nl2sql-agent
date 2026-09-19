#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation runner for the NL2SQL agent.

Runs the grounded evaluation suite against local test cases.
Supports mock (for CI) or real LLM (for production eval).
"""

from __future__ import annotations

import argparse
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql.agent import NL2SQLAgent
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb
from nl2sql.evaluate import load_golden, CaseOutcome, EvalReport
from nl2sql.llm import create_llm
from collections import Counter
import time
import statistics


def _norm_value(v):
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


def normalise_rows(rows):
    return Counter(tuple(_norm_value(v) for v in r) for r in rows)


def rows_match(candidate, reference):
    return normalise_rows(candidate) == normalise_rows(reference)


def preview(columns, rows, n=5):
    head = " | ".join(columns) if columns else "(no columns)"
    body = "\n".join(" | ".join(str(v) for v in r) for r in list(rows)[:n])
    more = f"\n... ({len(rows)} rows total)" if len(rows) > n else ""
    return f"{head}\n{body}{more}"


def evaluate_duckdb(agent, golden_cases, judge=None, progress=None):
    """Run evaluation using DuckDB for both reference and candidate queries."""
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    cases_out = []
    iterator = progress(golden_cases) if progress else golden_cases
    
    for g in iterator:
        # Run reference with DuckDB
        ref = run_query_duckdb(g.sql, data_dir=agent.executor_kwargs.get("data_dir", "./data"),
                               row_limit=agent.row_limit, timeout_s=agent.timeout_s)
        if not ref.ok:
            raise ValueError(f"Golden case {g.id} reference SQL failed: {ref.error}")
        
        res = agent.ask(g.question)
        exec_ok = res.status != "failed"
        match = exec_ok and rows_match(res.rows, ref.rows)
        judged = None
        
        if judge is not None and exec_ok and not match:
            verdict = judge.complete(
                judge_messages(g.question, g.sql, preview(ref.columns, ref.rows),
                               res.sql or "", preview(res.columns, res.rows))
            )
            judged = parse_judge(verdict)
        
        last = res.attempts[-1]
        error = "; ".join(last.guard_errors) or last.exec_error
        
        cases_out.append(CaseOutcome(
            id=g.id, question=g.question, status=res.status, exec_ok=exec_ok, match=match,
            judged=judged, repairs=res.repairs_used, confidence=res.confidence,
            total_ms=res.total_ms, candidate_sql=res.sql, reference_sql=g.sql,
            candidate_rows=len(res.rows), reference_rows=len(ref.rows), error=error, tags=g.tags
        ))
    
    return EvalReport(llm=getattr(agent.llm, "name", "?"), cases=cases_out, started=started,
                      elapsed_s=time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description="Run NL2SQL evaluation suite")
    parser.add_argument("--golden", "-g", default="golden/local_test.json",
                        help="Path to golden test cases JSON")
    parser.add_argument("--data-dir", "-d", default="./data",
                        help="Directory with CSV/Parquet/JSON files")
    parser.add_argument("--model", "-m", default="mock",
                        help="LLM model (mock, gpt-4o-mini, nemotron-3-ultra, etc.)")
    parser.add_argument("--judge-model", default=None,
                        help="Separate model for judge (e.g., gpt-4o)")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--row-limit", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-md", help="Save markdown report to file")
    parser.add_argument("--output-json", help="Save JSON report to file")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    # Load golden cases
    golden_cases = load_golden(args.golden)
    print(f"Loaded {len(golden_cases)} golden cases from {args.golden}")
    
    # Create schema
    schema = LocalFileSchema.from_data_dir(args.data_dir)
    if not schema.table_names():
        print(f"Error: No data files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded schema: {len(schema.table_names())} tables")
    
    # Create LLM
    if args.model == "mock":
        # Mock that returns golden SQL
        golden_map = {g.question: g.sql for g in golden_cases}
        
        class GoldenMockLLM:
            def __init__(self, golden_map):
                self.golden_map = golden_map
            def complete(self, messages):
                user_content = messages[-1]['content']
                for q, sql in self.golden_map.items():
                    if q in user_content:
                        return f"```sql\n{sql}\n```"
                return "```sql\nSELECT 1\n```"
        
        llm = GoldenMockLLM(golden_map)
        judge = None  # No judge for mock
    else:
        llm = create_llm(args.model)
        judge = create_llm(args.judge_model) if args.judge_model else None
    
    # Create agent with DuckDB executor
    agent = NL2SQLAgent(
        db_path=":memory:",
        llm=llm,
        schema=schema,
        max_repairs=args.max_repairs,
        row_limit=args.row_limit,
        timeout_s=args.timeout,
        executor=lambda db, sql, **kw: run_query_duckdb(sql, data_dir=args.data_dir, **kw),
    )
    
    # Run evaluation
    print(f"Running evaluation with model: {args.model}")
    report = evaluate_duckdb(agent, golden_cases, judge=judge)
    
    # Print results
    print(report.to_markdown())
    print()
    print("Summary:", report.summary())
    
    # Save outputs
    if args.output_md:
        Path(args.output_md).write_text(report.to_markdown(), encoding="utf-8")
        print(f"Saved markdown report to {args.output_md}")
    
    if args.output_json:
        import json
        Path(args.output_json).write_text(json.dumps({
            "summary": report.summary(),
            "cases": [asdict(c) for c in report.cases]
        }, indent=2, default=str), encoding="utf-8")
        print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()