"""
Query execution against SQLite in read-only mode with a row cap and a wall-clock timeout.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Sequence


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[Sequence[Any]]
    truncated: bool = False
    elapsed_ms: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def run_query(db_path: str, sql: str, *, row_limit: int = 200, timeout_s: float = 5.0) -> QueryResult:
    """Run one read-only query. Never raises: errors are returned in QueryResult.error."""
    t0 = time.perf_counter()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    timer = threading.Timer(timeout_s, conn.interrupt)
    try:
        timer.start()
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(row_limit + 1)
        truncated = len(rows) > row_limit
        return QueryResult(columns=columns, rows=rows[:row_limit], truncated=truncated,
                           elapsed_ms=int((time.perf_counter() - t0) * 1000))
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg.lower():
            msg = f"Query exceeded the {timeout_s:.0f}s time limit and was cancelled."
        return QueryResult(columns=[], rows=[], elapsed_ms=int((time.perf_counter() - t0) * 1000), error=msg)
    except sqlite3.Error as e:
        return QueryResult(columns=[], rows=[], elapsed_ms=int((time.perf_counter() - t0) * 1000), error=str(e))
    finally:
        timer.cancel()
        conn.close()
