#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for local NL2SQL testing with DuckDB backend.

Usage:
    python -m nl2sql.cli_local --question "Total revenue by country"
    python -m nl2sql.cli_local --question "Top customers" --data-dir ./my_data
"""

from __future__ import annotations

import argparse
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql.agent import NL2SQLAgent
from nl2sql.llm import create_llm
from nl2sql.schema_adapters import LocalFileSchema, create_schema
from nl2sql.executor_duckdb import run_query_duckdb


def main():
    parser = argparse.ArgumentParser(
        description="Local NL2SQL agent using DuckDB (BigQuery-compatible SQL)"
    )
    parser.add_argument("--question", "-q", required=True, help="Natural language question")
    parser.add_argument(
        "--data-dir", "-d", default="./data", help="Directory with CSV/Parquet/JSON files"
    )
    parser.add_argument(
        "--model", "-m", default="gpt-4o-mini", help="LLM model (OpenAI-compatible)"
    )
    parser.add_argument(
        "--max-repairs", type=int, default=2, help="Max repair attempts"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--trace-dir", help="Directory to save execution traces"
    )
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url", help="Custom OpenAI base URL (for local models)"
    )
    parser.add_argument(
        "--shots", type=int, default=4, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed trace"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show generated SQL without executing"
    )
    
    args = parser.parse_args()
    
    # Check data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        print("Create it with sample CSV/Parquet/JSON files.", file=sys.stderr)
        sys.exit(1)
    
    # Create schema from local files
    schema = LocalFileSchema.from_data_dir(args.data_dir)
    if not schema.table_names():
        print(f"Error: No valid data files found in {args.data_dir}", file=sys.stderr)
        print("Supported formats: .csv, .parquet, .json, .jsonl", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(schema.table_names())} tables: {', '.join(schema.table_names())}")
    if args.verbose:
        print(schema.card())
    
    # Create LLM
    if args.model == "mock":
        # Use OracleLLM which returns golden SQL for known questions
        from nl2sql.llm import OracleLLM
        llm = OracleLLM("golden/local_test.json")
    else:
        llm = create_llm(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    
    # Create agent with local schema and DuckDB executor
    agent = NL2SQLAgent(
        db_path=":memory:",
        llm=llm,
        schema=schema,
        max_repairs=args.max_repairs,
        confidence_threshold=args.confidence_threshold,
        trace_dir=args.trace_dir,
        shots=args.shots,
        executor=lambda db, sql, **kw: run_query_duckdb(sql, data_dir=args.data_dir, **kw),
    )
    
    try:
        if args.dry_run:
            # Just show the generation prompt
            from nl2sql.prompts import generation_messages
            card = schema.card()
            tables = schema.tables_mentioned(args.question, {})
            hints = schema.join_hints(tables)
            from nl2sql.examples import ExampleBank
            shots = ExampleBank([]).retrieve(args.question, k=args.shots)
            messages = generation_messages(args.question, card, hints, shots)
            print("=== Generation Prompt ===")
            for msg in messages:
                print(f"[{msg['role']}] {msg['content'][:500]}...")
            return
        
        # Run the agent
        print(f"\nQuestion: {args.question}")
        result = agent.ask(args.question)
        
        print(f"\nStatus: {result.status}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Repairs used: {result.repairs_used}")
        print(f"Total time: {result.total_ms}ms")
        
        if result.sql:
            print(f"\nGenerated SQL:\n{result.sql}")
        
        if result.status == "ok" and result.rows:
            print(f"\nResults ({len(result.rows)} rows):")
            print(" | ".join(result.columns))
            print("-" * 80)
            for row in result.rows[:20]:
                print(" | ".join(str(v) for v in row))
            if result.truncated:
                print(f"... ({len(result.rows)} rows total, showing first 20)")
        elif result.status == "failed" and result.attempts:
            last = result.attempts[-1]
            error = "; ".join(last.guard_errors) or last.exec_error
            if error:
                print(f"\nError: {error}")
        
        if args.verbose and result.attempts:
            print("\n=== Attempt Trace ===")
            for i, attempt in enumerate(result.attempts):
                print(f"\nAttempt {i+1} ({attempt.kind}):")
                print(f"  SQL: {attempt.sql}")
                if attempt.guard_errors:
                    print(f"  Guard errors: {attempt.guard_errors}")
                if attempt.exec_error:
                    print(f"  Exec error: {attempt.exec_error}")
                print(f"  Rows: {attempt.row_count}, LLM: {attempt.llm_ms}ms, Exec: {attempt.exec_ms}ms")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()