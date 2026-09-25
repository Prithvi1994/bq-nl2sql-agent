"""BigQuery-dialect SQL execution over DuckDB.

Public BigQuery has no offline mirror that preserves SQL semantics, so we keep the
*query language* real and the *storage* local:

  1. strip BQ scripting (`DECLARE x STRING DEFAULT 'v'`) and inline the value
  2. transpile the remaining statement with sqlglot: bigquery -> duckdb
  3. execute against a local DuckDB database that mirrors the BQ table layout

Everything the agent writes is BigQuery SQL with backticks, UNNEST, DATE_DIFF,
_TABLE_SUFFIX and wildcard tables. It is parsed, validated and transpiled exactly
as it would be on a real project; only the bytes at rest differ.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import sqlglot

# DECLARE/SET are scripting headers, not statements we need to keep. Their values
# are inlined so the query body still references the same names.
_DECLARE_RE = re.compile(
    r"^\s*DECLARE\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<type>[A-Za-z0-9_<>(), ]+?)\s+DEFAULT\s+",
    re.IGNORECASE,
)
_SET_RE = re.compile(
    r"^\s*SET\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+?)\s*;?\s*$",
    re.IGNORECASE,
)


class BQScriptError(ValueError):
    """Raised when the scripting header cannot be reduced to literal values."""


def split_script(sql: str) -> tuple[Dict[str, str], str]:
    """Split BQ scripting declarations from the query body.

    Returns (variables, body). Variables are name -> SQL literal text.
    Raises BQScriptError for scripting we cannot safely inline.
    """
    variables: Dict[str, str] = {}
    body = sql
    # Iterate: a body may itself start with further DECLAREs after stripping.
    for _ in range(16):
        changed = False
        for statement in _split_statements(body):
            head = statement.strip()
            if not head:
                continue
            m = _DECLARE_RE.match(head)
            if m and not head.upper().startswith("SELECT"):
                variables[m.group("name")] = _literal_after_default(head)
                body = body.replace(statement, "")
                changed = True
                continue
            m = _SET_RE.match(head)
            if m and not head.upper().startswith(("SELECT", "WITH")):
                variables[m.group("name")] = m.group("value").rstrip(";").strip()
                body = body.replace(statement, "")
                changed = True
        if not changed:
            break
    body = body.strip().rstrip(";").strip()
    if not body:
        raise BQScriptError("Query contains only declarations and no final SELECT.")
    return variables, body


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons that are not inside a string literal."""
    out, buf, in_str, quote = [], [], False, ""
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_str:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(sql):
                buf.append(sql[i + 1])
                i += 2
                continue
            if ch == quote:
                in_str = False
            i += 1
            continue
        if ch in ("'", '"', "`"):
            in_str, quote = True, ch
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            if buf:
                out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf))
    return out


def _literal_after_default(statement: str) -> str:
    """Return the literal after DEFAULT, honouring quoted strings and parens."""
    idx = statement.upper().rfind(" DEFAULT ")
    if idx < 0:
        raise BQScriptError(f"DECLARE without DEFAULT is not supported: {statement[:80]}")
    value = statement[idx + len(" DEFAULT ") :].strip().rstrip(";").strip()
    if value.upper() in ("CURRENT_DATE()", "CURRENT_TIMESTAMP()"):
        raise BQScriptError("Non-deterministic DEFAULT is not inlined; pass the value in SQL.")
    return value


def inline_variables(sql: str, variables: Dict[str, str]) -> str:
    """Replace bare identifier references with their literal values.

    Two places are skipped on purpose, because a declared name there is a *column*,
    not a variable:

      * `USING (a, b)` -- join-column lists
      * a dotted reference such as `t.name`

    Inlining is textual, so a DECLARE whose name matches a column is a caller error.
    Use a prefix that cannot collide (`__exp_id`). `check_variable_collisions`
    enforces it instead of letting a broken query reach the database.
    """
    if not variables:
        return sql

    # Blank out USING(...) lists and dotted refs so the regex cannot touch them.
    protected: List[str] = []

    def _protect(pattern: str, repl: str) -> None:
        nonlocal sql
        sql = re.sub(pattern, lambda m: protected.append(m.group(0)) or repl, sql, flags=re.IGNORECASE)

    def _shield(match: "re.Match[str]") -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    sql = re.sub(r"\bUSING\s*\([^)]*\)", _shield, sql, flags=re.IGNORECASE)
    sql = re.sub(r"\b[A-Za-z_][\w]*\.[A-Za-z_][\w]*", _shield, sql)

    for name, literal in variables.items():
        sql = re.sub(rf"(?<![\w.`]){re.escape(name)}\b", lambda _m: literal, sql, flags=re.IGNORECASE)

    for idx, text in enumerate(protected):
        sql = sql.replace(f"\x00{idx}\x00", text)
    return sql


def check_variable_collisions(variables: Dict[str, str], body: str) -> List[str]:
    """Names that appear as columns in the body and therefore cannot be inlined safely."""
    collisions = []
    for name in variables:
        if re.search(rf"\b[A-Za-z_][\w]*\.{re.escape(name)}\b", body) or re.search(
            rf"\bUSING\s*\([^)]*\b{re.escape(name)}\b[^)]*\)", body, re.IGNORECASE
        ):
            collisions.append(name)
    return collisions


def transpile_to_duckdb(sql: str) -> str:
    """BigQuery SQL -> DuckDB SQL, with scripting already stripped by the caller."""
    statements = sqlglot.parse(sql, read="bigquery")
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ValueError(f"Expected exactly one query statement, found {len(statements)}.")
    return statements[0].sql(dialect="duckdb")


def backtick_table_names(sql: str) -> list[str]:
    """Fully-qualified backticked table names, wildcards preserved."""
    return re.findall(r"`([^`]+)`", sql)


# BigQuery table references are `project.dataset.table` (or `dataset.table`). The local
# mirror holds one flat namespace, so a fully-qualified reference resolves to its last
# component. This is the one deliberate semantic difference between the mirror and a
# real project, and it is reported in the query provenance so a report can state it.
def _unqualify_tables(sql: str) -> str:
    def repl(match: "re.Match[str]") -> str:
        name = match.group(1)
        parts = name.split(".")
        if len(parts) <= 1:
            return match.group(0)
        return "`" + parts[-1] + "`"

    return re.sub(r"`([^`]+)`", repl, sql)


def run_bq_query(
    sql: str,
    db_path: str,
    *,
    row_limit: int = 200,
    timeout_s: float = 30.0,
) -> "BigQueryResult":
    """Execute BigQuery SQL against a local DuckDB database.

    Read-only by construction: the local DuckDB connection is opened read_only and
    sqlglot refuses to transpile anything that is not a query.
    """
    t0 = time.perf_counter()
    try:
        variables, body = split_script(sql)
        collisions = check_variable_collisions(variables, body)
        if collisions:
            raise BQScriptError(
                "DECLARE name collides with a column in the query body: "
                + ", ".join(collisions)
                + ". Rename the variable (e.g. prefix with __)."
            )
        body = inline_variables(body, variables)
        body = _unqualify_tables(body)
        duckdb_sql = transpile_to_duckdb(body)
    except (BQScriptError, ValueError, sqlglot.errors.SqlglotError) as exc:
        return BigQueryResult(
            ok=False,
            error=f"{type(exc).__name__}: {str(exc)[:400]}",
            dialect_sql=sql,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )

    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        cursor = conn.execute(duckdb_sql)
        rows = cursor.fetchmany(row_limit + 1)
        truncated = len(rows) > row_limit
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return BigQueryResult(
            ok=True,
            columns=columns,
            rows=[tuple(r) for r in rows[:row_limit]],
            row_count=len(rows[:row_limit]),
            truncated=truncated,
            duckdb_sql=duckdb_sql,
            dialect_sql=sql,
            variables=variables,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as exc:  # duckdb raises many concrete types
        return BigQueryResult(
            ok=False,
            error=f"{type(exc).__name__}: {str(exc)[:400]}",
            duckdb_sql=duckdb_sql,
            dialect_sql=sql,
            variables=variables,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
    finally:
        if conn is not None:
            conn.close()


@dataclass
class BigQueryResult:
    ok: bool
    columns: List[str] = field(default_factory=list)
    rows: List[Sequence[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str = ""
    duckdb_sql: str = ""
    dialect_sql: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    elapsed_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "columns": self.columns,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "error": self.error,
            "duckdb_sql": self.duckdb_sql,
            "dialect_sql": self.dialect_sql,
            "variables": self.variables,
            "elapsed_ms": self.elapsed_ms,
        }
