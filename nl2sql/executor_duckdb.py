#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DuckDB executor for local BigQuery-compatible SQL testing.

Queries local CSV/Parquet/JSON files as tables using BigQuery-compatible syntax.
No cloud credentials or project required.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Any, List, Sequence

import duckdb


@dataclass
class QueryResult:
    """Result of a query execution."""
    ok: bool
    columns: List[str]
    rows: List[Sequence[Any]]
    truncated: bool
    error: str = ""
    elapsed_ms: int = 0
    bytes_processed: int = 0  # DuckDB doesn't expose this easily; placeholder


def register_data_files(conn: duckdb.DuckDBPyConnection, data_dir: str) -> int:
    """
    Auto-register all supported data files in data_dir as views/tables.
    
    Supported formats: .csv, .parquet, .json, .jsonl
    Table name = filename without extension.
    """
    if not os.path.isdir(data_dir):
        return 0
    
    registered = 0
    for filepath in glob.glob(os.path.join(data_dir, "*")):
        if not os.path.isfile(filepath):
            continue
        table_name = os.path.splitext(os.path.basename(filepath))[0]
        # Sanitize table name for SQL
        table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
        if not table_name:
            continue
        
        try:
            if filepath.endswith(".csv"):
                conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
            elif filepath.endswith(".parquet"):
                conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{filepath}')")
            elif filepath.endswith(".json") or filepath.endswith(".jsonl"):
                conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_json_auto('{filepath}')")
            else:
                continue
            registered += 1
        except Exception:
            # Silently skip files that can't be read
            pass
    
    return registered


def run_query_duckdb(
    sql: str,
    data_dir: str = "./data",
    row_limit: int = 200,
    timeout_s: float = 30.0,
) -> QueryResult:
    """
    Execute BigQuery-compatible SQL against local data files using DuckDB.
    
    Args:
        sql: SQL query (BigQuery dialect; DuckDB supports most BQ syntax)
        data_dir: Directory containing CSV/Parquet/JSON files
        row_limit: Maximum rows to return (safety limit)
        timeout_s: Query timeout in seconds
    
    Returns:
        QueryResult with ok, columns, rows, error, timing
    """
    t0 = time.perf_counter()
    conn = duckdb.connect()
    conn.execute(f"SET enable_progress_bar = false")
    conn.execute(f"SET threads = 4")
    
    try:
        # Register local data files as views
        registered = register_data_files(conn, data_dir)
        if registered == 0:
            return QueryResult(
                ok=False,
                columns=[],
                rows=[],
                truncated=False,
                error=f"No data files found in {data_dir}. Add CSV/Parquet/JSON files.",
                elapsed_ms=int((time.perf_counter() - t0) * 1000)
            )
        
        # Safety: enforce LIMIT if not present
        sql_stripped = sql.strip().rstrip(";")
        has_limit = "limit" in sql_stripped.lower()
        # Also check for LIMIT in CTEs/subqueries - simple heuristic
        if not has_limit:
            sql_stripped = f"{sql_stripped} LIMIT {row_limit}"
        
        # Execute
        result = conn.execute(sql_stripped).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []
        
        return QueryResult(
            ok=True,
            columns=columns,
            rows=result,
            truncated=len(result) >= row_limit,
            elapsed_ms=int((time.perf_counter() - t0) * 1000)
        )
    except Exception as e:
        return QueryResult(
            ok=False,
            columns=[],
            rows=[],
            truncated=False,
            error=str(e),
            elapsed_ms=int((time.perf_counter() - t0) * 1000)
        )
    finally:
        conn.close()


def run_query_duckdb_with_context(
    sql: str,
    data_dir: str = "./data",
    row_limit: int = 200,
    timeout_s: float = 30.0,
    extra_views: dict[str, str] = None,
) -> QueryResult:
    """
    Extended version allowing additional ad-hoc views.
    
    Args:
        extra_views: Dict of {view_name: SELECT ...} to register before query
    """
    t0 = time.perf_counter()
    conn = duckdb.connect()
    conn.execute(f"SET enable_progress_bar = false")
    
    try:
        register_data_files(conn, data_dir)
        
        # Register extra views
        if extra_views:
            for name, view_sql in extra_views.items():
                conn.execute(f"CREATE OR REPLACE VIEW {name} AS {view_sql}")
        
        sql_stripped = sql.strip().rstrip(";")
        if "limit" not in sql_stripped.lower():
            sql_stripped = f"{sql_stripped} LIMIT {row_limit}"
        
        result = conn.execute(sql_stripped).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []
        
        return QueryResult(
            ok=True,
            columns=columns,
            rows=result,
            truncated=len(result) >= row_limit,
            elapsed_ms=int((time.perf_counter() - t0) * 1000)
        )
    except Exception as e:
        return QueryResult(
            ok=False,
            columns=[],
            rows=[],
            truncated=False,
            error=str(e),
            elapsed_ms=int((time.perf_counter() - t0) * 1000)
        )
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick self-test
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV
        test_csv = os.path.join(tmpdir, "users.csv")
        with open(test_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "value"])
            writer.writerow([1, "alice", 100])
            writer.writerow([2, "bob", 200])
        
        # Test query
        result = run_query_duckdb("SELECT name, SUM(value) FROM users GROUP BY name", data_dir=tmpdir)
        print(f"OK: {result.ok}")
        print(f"Columns: {result.columns}")
        print(f"Rows: {result.rows}")
        print(f"Error: {result.error}")