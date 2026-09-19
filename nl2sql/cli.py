#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified CLI for NL2SQL agent with multiple execution modes.

Usage:
    # Local mode (DuckDB with local files)
    python -m nl2sql.cli --mode local --question "Total revenue by country" --data-dir ./data
    
    # BigQuery mode (remote)
    python -m nl2sql.cli --mode bigquery --question "Total revenue by country" \
        --project my-project --dataset my_dataset
    
    # BigQuery with local data (hybrid: BQ schema, DuckDB execution)
    python -m nl2sql.cli --mode hybrid --question "Total revenue by country" \
        --project my-project --dataset my_dataset --data-dir ./data
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Dict, Any, List

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql.agent import NL2SQLAgent
from nl2sql.llm import create_llm, OracleLLM
from nl2sql.schema_adapters import create_schema, LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb


def run_query_bigquery(db_path: str, sql: str, row_limit: int = 200,
                       timeout_s: float = 30.0, **kwargs) -> Any:
    """Execute query on BigQuery."""
    from google.cloud import bigquery
    from nl2sql.executor import QueryResult
    
    project = kwargs.get("project")
    location = kwargs.get("location", "US")
    credentials = kwargs.get("credentials")
    
    if not project:
        return QueryResult(ok=False, error="BigQuery project not configured", 
                          columns=[], rows=[], elapsed_ms=0, truncated=False)
    
    client = bigquery.Client(project=project, credentials=credentials, location=location)
    
    import time
    t0 = time.perf_counter()
    try:
        job = client.query(sql)
        result = job.result(timeout=timeout_s)
        rows = [tuple(row) for row in result]
        columns = [field.name for field in result.schema]
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        
        truncated = len(rows) > row_limit
        if truncated:
            rows = rows[:row_limit]
        
        return QueryResult(ok=True, error=None, columns=columns, rows=rows,
                          elapsed_ms=elapsed_ms, truncated=truncated)
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return QueryResult(ok=False, error=str(e), columns=[], rows=[],
                          elapsed_ms=elapsed_ms, truncated=False)


def build_agent(mode: str, args) -> NL2SQLAgent:
    """Build agent based on mode."""
    
    # Create LLM
    if args.model == "mock":
        llm = OracleLLM("golden/local_test.json")
    else:
        llm = create_llm(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    
    # Create schema and executor based on mode
    if mode == "local":
        # Local DuckDB mode
        if not os.path.isdir(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        
        schema = LocalFileSchema.from_data_dir(args.data_dir)
        if not schema.table_names():
            print(f"Error: No valid data files found in {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        
        executor = lambda db, sql, **kw: run_query_duckdb(sql, data_dir=args.data_dir, **kw)
        db_path = ":memory:"
        
    elif mode == "bigquery":
        # BigQuery remote mode
        if not args.project or not args.dataset:
            print("Error: BigQuery mode requires --project and --dataset", file=sys.stderr)
            sys.exit(1)
        
        schema = create_schema(
            "bigquery",
            project=args.project,
            dataset=args.dataset,
            credentials=args.credentials,
            location=args.location,
            include_profiles=args.include_profiles,
            include_policy_tags=args.include_policy_tags,
            include_partitions=args.include_partitions,
            include_clustering=args.include_clustering,
            include_storage=args.include_storage,
        )
        
        executor = run_query_bigquery
        db_path = f"{args.project}.{args.dataset}"
        
    elif mode == "hybrid":
        # Hybrid: BigQuery schema, local DuckDB execution
        if not args.project or not args.dataset:
            print("Error: Hybrid mode requires --project and --dataset", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        
        # Load schema from BigQuery
        schema = create_schema(
            "bigquery",
            project=args.project,
            dataset=args.dataset,
            credentials=args.credentials,
            location=args.location,
            include_profiles=args.include_profiles,
            include_policy_tags=args.include_policy_tags,
            include_partitions=args.include_partitions,
            include_clustering=args.include_clustering,
            include_storage=args.include_storage,
        )
        
        # Execute locally with DuckDB
        executor = lambda db, sql, **kw: run_query_duckdb(sql, data_dir=args.data_dir, **kw)
        db_path = ":memory:"
        
    else:
        print(f"Error: Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)
    
    # Print schema info
    print(f"Mode: {mode}")
    print(f"Tables: {', '.join(schema.table_names())}")
    if args.verbose:
        print(schema.card())
        # Print BQ metadata if available
        if hasattr(schema, 'get_enriched_schema'):
            enriched = schema.get_enriched_schema()
            for table_name, info in enriched.items():
                if info.get('partitions'):
                    print(f"  {table_name} partitions: {info['partitions']}")
                if info.get('clustering'):
                    print(f"  {table_name} clustering: {info['clustering']}")
    
    # Create agent
    agent = NL2SQLAgent(
        db_path=db_path,
        llm=llm,
        schema=schema,
        max_repairs=args.max_repairs,
        confidence_threshold=args.confidence_threshold,
        trace_dir=args.trace_dir,
        shots=args.shots,
        executor=executor,
        executor_kwargs={
            "project": args.project,
            "location": args.location,
            "credentials": args.credentials,
        } if mode in ("bigquery", "hybrid") else {},
    )
    
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="NL2SQL Agent - Multiple execution modes"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", choices=["local", "bigquery", "hybrid"], default="local",
        help="Execution mode: local (DuckDB), bigquery (remote), hybrid (BQ schema + local exec)"
    )
    
    # Question
    parser.add_argument("--question", "-q", required=True, help="Natural language question")
    
    # Local mode args
    parser.add_argument(
        "--data-dir", "-d", default="./data",
        help="Directory with CSV/Parquet/JSON files (local/hybrid mode)"
    )
    
    # BigQuery mode args
    parser.add_argument("--project", help="GCP project ID (bigquery/hybrid mode)")
    parser.add_argument("--dataset", help="BigQuery dataset ID (bigquery/hybrid mode)")
    parser.add_argument("--location", default="US", help="BigQuery location")
    parser.add_argument("--credentials", help="Path to GCP credentials JSON")
    
    # BigQuery metadata options
    parser.add_argument("--include-profiles", action="store_true", default=True,
                        help="Include column profiles (histograms, null rates)")
    parser.add_argument("--no-profiles", action="store_false", dest="include_profiles",
                        help="Disable column profiles")
    parser.add_argument("--include-policy-tags", action="store_true", default=True,
                        help="Include policy tags")
    parser.add_argument("--no-policy-tags", action="store_false", dest="include_policy_tags",
                        help="Disable policy tags")
    parser.add_argument("--include-partitions", action="store_true", default=True,
                        help="Include partitioning info")
    parser.add_argument("--no-partitions", action="store_false", dest="include_partitions",
                        help="Disable partitioning info")
    parser.add_argument("--include-clustering", action="store_true", default=True,
                        help="Include clustering info")
    parser.add_argument("--no-clustering", action="store_false", dest="include_clustering",
                        help="Disable clustering info")
    parser.add_argument("--include-storage", action="store_true", default=True,
                        help="Include storage info")
    parser.add_argument("--no-storage", action="store_false", dest="include_storage",
                        help="Disable storage info")
    
    # LLM args
    parser.add_argument("--model", "-m", default="nemotron-3-ultra", help="LLM model")
    parser.add_argument("--api-key", help="OpenRouter/OpenAI API key")
    parser.add_argument("--base-url", help="Custom base URL (for local models)")
    parser.add_argument("--shots", type=int, default=4, help="Few-shot examples")
    
    # Agent args
    parser.add_argument("--max-repairs", type=int, default=2, help="Max repair attempts")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--trace-dir", help="Directory to save execution traces")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed trace")
    parser.add_argument("--dry-run", action="store_true", help="Show generated SQL without executing")
    
    args = parser.parse_args()
    
    # Build agent
    agent = build_agent(args.mode, args)
    
    # Run
    try:
        if args.dry_run:
            from nl2sql.prompts import generation_messages
            from nl2sql.examples import ExampleBank
            card = agent.schema.card()
            tables = agent.schema.tables_mentioned(args.question, agent.synonyms)
            hints = agent.schema.join_hints(tables)
            shots = agent.examples.retrieve(args.question, k=args.shots)
            messages = generation_messages(args.question, card, hints, shots)
            print("=== Generation Prompt ===")
            for msg in messages:
                print(f"[{msg['role']}] {msg['content'][:500]}...")
            return
        
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