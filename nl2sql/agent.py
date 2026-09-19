"""
The agent: ground → generate → guard → execute, with a bounded repair loop and
an evidence trace for every attempt.

Confidence is deliberately simple and honest: it reflects how much repair the
query needed, not the model's opinion of itself. Anything below the threshold
is returned as needs_review rather than presented as an answer.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .examples import ExampleBank
from .executor import QueryResult, run_query
from .guard import SQLGuardError, validate
from .llm import LLM
from .prompts import extract_sql, generation_messages, repair_messages
from .schema import Schema

CONFIDENCE_BY_REPAIRS = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3}


@dataclass
class Attempt:
    kind: str                      # "generate" | "repair"
    raw_reply: str
    sql: Optional[str]
    guard_errors: List[str] = field(default_factory=list)
    exec_error: str = ""
    row_count: int = 0
    llm_ms: int = 0
    exec_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return bool(self.sql) and not self.guard_errors and not self.exec_error


@dataclass
class AskResult:
    trace_id: str
    question: str
    status: str                    # "ok" | "needs_review" | "failed"
    sql: Optional[str]
    columns: List[str]
    rows: List[Sequence[Any]]
    truncated: bool
    repairs_used: int
    confidence: float
    total_ms: int
    attempts: List[Attempt]
    tables_hinted: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rows"] = [list(r) for r in self.rows[:20]]  # keep traces small
        return d


class NL2SQLAgent:
    def __init__(self, db_path: str, llm: LLM, *, examples: Optional[ExampleBank] = None,
                 max_repairs: int = 2, row_limit: int = 200, timeout_s: float = 5.0,
                 confidence_threshold: float = 0.5, trace_dir: Optional[str] = None,
                 shots: int = 4, synonyms: Optional[Dict[str, Sequence[str]]] = None,
                 schema: Optional[Schema] = None,
                 executor: Optional[callable] = None,
                 executor_kwargs: Optional[dict] = None):
        self.db_path = db_path
        self.llm = llm
        self.schema = schema or Schema.from_sqlite(db_path)
        self.examples = examples or ExampleBank([])
        self.max_repairs = max_repairs
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        self.confidence_threshold = confidence_threshold
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.shots = shots
        self.synonyms = synonyms or {}
        self._executor = executor or run_query
        self.executor_kwargs = executor_kwargs or {}

    # ------------------------------------------------------------------
    def _call(self, messages) -> tuple[str, int]:
        t0 = time.perf_counter()
        reply = self.llm.complete(messages)
        return reply, int((time.perf_counter() - t0) * 1000)

    def _check(self, sql: Optional[str]) -> List[str]:
        if not sql:
            return ["No SQL query was found in the reply. Return the query inside a ```sql block."]
        try:
            _, errors = validate(sql, self.schema)
            return errors
        except SQLGuardError as e:
            return [str(e)]

    # ------------------------------------------------------------------
    def ask(self, question: str) -> AskResult:
        t_start = time.perf_counter()
        card = self.schema.card()
        tables = self.schema.tables_mentioned(question, self.synonyms)
        shots = self.examples.retrieve(question, k=self.shots)
        if not tables:
            # Nothing named explicitly: borrow the tables the most similar examples used. When the question
            # does name tables, hints stay restricted to those, because extra hints invite spurious joins.
            for ex in shots:
                for name in self.schema.table_names():
                    if name.lower() in ex.sql.lower() and name not in tables:
                        tables.append(name)
        hints = self.schema.join_hints(tables)

        attempts: List[Attempt] = []
        reply, llm_ms = self._call(generation_messages(question, card, hints, shots))
        sql = extract_sql(reply)
        attempt = Attempt(kind="generate", raw_reply=reply, sql=sql, llm_ms=llm_ms)
        attempts.append(attempt)

        result: Optional[QueryResult] = None
        repairs = 0
        while True:
            attempt.guard_errors = self._check(attempt.sql)
            if not attempt.guard_errors:
                result = self._executor(self.db_path, attempt.sql, row_limit=self.row_limit, timeout_s=self.timeout_s, **self.executor_kwargs)
                attempt.exec_error = result.error
                attempt.exec_ms = result.elapsed_ms
                attempt.row_count = len(result.rows)
                if result.ok:
                    break
            if repairs >= self.max_repairs:
                break
            repairs += 1
            problems = attempt.guard_errors or [f"SQLite error: {attempt.exec_error}"]
            reply, llm_ms = self._call(repair_messages(question, card, hints, attempt.sql or "", problems))
            attempt = Attempt(kind="repair", raw_reply=reply, sql=extract_sql(reply), llm_ms=llm_ms)
            attempts.append(attempt)

        final = attempts[-1]
        if result is not None and result.ok and final.succeeded:
            confidence = CONFIDENCE_BY_REPAIRS.get(repairs, 0.2)
            status = "ok" if confidence >= self.confidence_threshold else "needs_review"
            out = AskResult(trace_id=uuid.uuid4().hex[:12], question=question, status=status, sql=final.sql,
                            columns=result.columns, rows=result.rows, truncated=result.truncated,
                            repairs_used=repairs, confidence=confidence,
                            total_ms=int((time.perf_counter() - t_start) * 1000), attempts=attempts,
                            tables_hinted=tables)
        else:
            out = AskResult(trace_id=uuid.uuid4().hex[:12], question=question, status="failed", sql=final.sql,
                            columns=[], rows=[], truncated=False, repairs_used=repairs, confidence=0.0,
                            total_ms=int((time.perf_counter() - t_start) * 1000), attempts=attempts,
                            tables_hinted=tables)
        self._trace(out)
        return out

    # ------------------------------------------------------------------
    def _trace(self, result: AskResult) -> None:
        if not self.trace_dir:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "llm": getattr(self.llm, "name", "?"), **result.to_dict()}
        with (self.trace_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
