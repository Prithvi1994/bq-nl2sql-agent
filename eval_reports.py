#!/usr/bin/env python
"""
Simple report evaluation script using deepseek-v4-flash for grading.
Run: python eval_reports.py --golden golden/reports.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql.llm import create_llm
from nl2sql.prompts import grade_report, GraderResult
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb
from nl2sql.agent import NL2SQLAgent
from nl2sql.report_graph import run_report_generation, ReportStyle


def load_golden(path: str):
    """Load golden test cases from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def run_grader(
    question: str,
    report: str,
    sql: str,
    columns: list,
    rows: list,
    style: str,
    model: str = "deepseek/deepseek-v4-flash-0731:free",
) -> dict:
    """Run the grader on a single report."""
    from nl2sql.prompts import grade_report
    from nl2sql.llm import create_llm
    
    judge_llm = create_llm("deepseek/deepseek-v4-flash-0731:free")
    return grade_report(
        question=question,
        report=report,
        sql=sql,
        columns=columns,
        rows=rows,
        style=style,
        model="deepseek/deepseek-v4-flash-0731:free",
    )


def run_one_case(agent, case, data_dir, judge_model="deepseek/deepseek-v4-flash-0731:free"):
    """Run a single test case and return results."""
    question = case["question"]
    expected_sql = case["sql"]
    style = case.get("style", "executive_summary")
    tags = case.get("tags", [])
    case_id = case.get("id", "unknown")
    
    start = time.time()
    
    try:
        # Generate report
        report_result = run_report_generation(
            question=case["question"],
            style=case.get("style", "executive_summary"),
            model="nemotron-3-ultra",
            data_dir=".",
            enable_human_review=False,
        )
        
        # Build report text
        report_text = report_result.get("summary", "") + "\n\n" + "\n\n".join(
            f"## {s['title']}\n{s['content']}" for s in report_result.get("sections", [])
        )
        
        # Run reference query
        ref_result = run_query_duckdb(
            case["sql"], data_dir=".", row_limit=100, timeout_s=30
        )
        
        if not ref_result.ok:
            return {
                "id": case.get("id"),
                "question": case["question"],
                "error": f"Reference SQL failed: {ref_result.error}",
                "execution_time": time.time() - start,
                "tags": tags,
            }
        
        # Grade the report (use already-generated report, no second LLM call)
        grader_result = grade_report(
            question=case["question"],
            report=report_text,
            sql=case["sql"],
            columns=ref_result.columns,
            rows=ref_result.rows,
            style=style,
            model="deepseek/deepseek-v4-flash-0731:free",
        )
        
        return {
            "id": case.get("id"),
            "question": case["question"],
            "expected_sql": case["sql"],
            "grader_decision": grader_result.decision,
            "grader_scores": grader_result.scores,
            "grader_feedback": grader_result.feedback,
            "grader_overall": grader_result.overall,
            "execution_time": time.time() - start,
            "tags": tags,
        }
        
    except Exception as e:
        return {
            "id": case.get("id"),
            "question": case["question"],
            "error": str(e),
            "execution_time": time.time() - start,
            "tags": tags,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate report generation quality")
    parser.add_argument("--golden", default="golden/reports.json", help="Path to golden test cases")
    parser.add_argument("--data-dir", default=".", help="Data directory")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4-flash-0731:free", help="Judge model")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if not os.path.exists(args.golden):
        print(f"Error: Golden file not found: {args.golden}")
        sys.exit(1)
    
    golden_cases = load_golden(args.golden)
    
    # Setup agent
    schema = LocalFileSchema.from_data_dir(".")
    agent = NL2SQLAgent(
        db_path=":memory:",
        llm=create_llm("nemotron-3-ultra"),
        schema=schema,
        executor=lambda db, sql, **kw: run_query_duckdb(sql, data_dir=".", **kw),
    )
    
    print(f"Evaluating {len(golden_cases)} cases with deepseek-v4-flash judge...")
    print("-" * 60)
    
    results = []
    total_start = time.time()
    
    for case in golden_cases:
        start = time.time()
        result = run_one_case(agent, case, ".", "deepseek/deepseek-v4-flash-0731:free")
        result["execution_time"] = time.time() - start
        results.append(result)
        
        if args.verbose:
            decision = result.get("grader_decision", "error")
            overall = result.get("grader_overall", 0)
            print(f"  [{result['id']}] {decision} (score: {result.get('grader_overall', 0):.2f}) - {time.time()-start:.1f}s")
    
    total_time = time.time() - total_start
    
    # Summary
    total = len(results)
    approved = sum(1 for r in results if r.get("grader_decision") == "approve")
    revised = sum(1 for r in results if r.get("grader_decision") == "revise")
    rejected = sum(1 for r in results if r.get("grader_decision") == "reject")
    errors = sum(1 for r in results if "error" in r)
    avg_score = sum(r.get("grader_overall", 0) for r in results if "grader_overall" in r) / max(1, len([r for r in results if "grader_overall" in r]))
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total cases:      {len(results)}")
    print(f"Approved:         {approved}")
    print(f"Revised:          {revised}")
    print(f"Rejected:         {rejected}")
    print(f"Errors:           {errors}")
    print(f"Avg score:        {avg_score:.2f}")
    print(f"Total time:       {time.time() - total_start:.1f}s")
    print("=" * 60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "summary": {
                    "total": len(results),
                    "approved": approved,
                    "revised": revised,
                    "rejected": rejected,
                    "errors": errors,
                    "avg_score": round(avg_score, 3),
                    "total_time": round(time.time() - total_start, 2),
                },
                "results": results
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()