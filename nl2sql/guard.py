"""
SQL guard: parse with sqlglot, refuse anything that is not a single read-only
statement, and check every referenced table and column against the schema
before the database ever sees the query. Errors are returned as plain
sentences so they can be fed straight back to the model for repair.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from .schema import Schema

_FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.TruncateTable,
    exp.Command, exp.Merge, exp.Pragma, exp.Transaction, exp.Commit, exp.Rollback,
)


class SQLGuardError(Exception):
    """Raised for SQL that must never run: unparsable, multi-statement, or not read-only."""


def parse(sql: str) -> exp.Expression:
    """Parse one SQLite statement. Multiple statements are rejected."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.SqlglotError as e:  # ParseError, TokenError (unterminated strings, stray characters)...
        raise SQLGuardError(f"SQL does not parse: {str(e).splitlines()[0][:200]}") from e
    statements = [s for s in statements if s is not None]
    if not statements:
        raise SQLGuardError("No SQL statement found.")
    if len(statements) > 1:
        raise SQLGuardError(f"Expected one statement, found {len(statements)}.")
    return statements[0]


def assert_read_only(expr: exp.Expression) -> None:
    if not isinstance(expr, (exp.Select, exp.Union, exp.Subquery)):
        raise SQLGuardError(f"Only SELECT queries are allowed, got {type(expr).__name__.upper()}.")
    for node in expr.walk():
        if isinstance(node, _FORBIDDEN):
            raise SQLGuardError(f"Statement contains a forbidden {type(node).__name__.upper()} clause.")


def _alias_map(expr: exp.Expression) -> Tuple[Dict[str, str], Set[str]]:
    """Map every alias or bare name used in FROM/JOIN to its real table; collect CTE/subquery aliases."""
    aliases: Dict[str, str] = {}
    virtual: Set[str] = set()
    for cte in expr.find_all(exp.CTE):
        virtual.add(cte.alias_or_name.lower())
    for sub in expr.find_all(exp.Subquery):
        if sub.alias:
            virtual.add(sub.alias.lower())
    for t in expr.find_all(exp.Table):
        real = t.name
        key = (t.alias or t.name).lower()
        aliases[key] = real
    return aliases, virtual


def validate_against_schema(expr: exp.Expression, schema: Schema) -> List[str]:
    """Return human-readable errors for unknown tables and columns; [] when clean."""
    errors: List[str] = []
    aliases, virtual = _alias_map(expr)
    # Aliases defined in a select list ("SUM(x) AS revenue") are legal in ORDER BY / GROUP BY / HAVING.
    select_aliases = {a.alias.lower() for a in expr.find_all(exp.Alias) if a.alias}

    real_tables: List[str] = []
    for t in expr.find_all(exp.Table):
        if t.name.lower() in virtual:
            continue
        canonical = schema.resolve_table(t.name)
        if not canonical:
            errors.append(f"Unknown table '{t.name}'. Available tables: {', '.join(schema.table_names())}.")
        elif canonical not in real_tables:
            real_tables.append(canonical)

    for c in expr.find_all(exp.Column):
        name = c.name
        if not name or name == "*":
            continue
        qualifier = (c.table or "").lower()
        if qualifier:
            if qualifier in virtual:
                continue
            real = aliases.get(qualifier) or qualifier
            canonical = schema.resolve_table(real)
            if not canonical:
                continue  # already reported as unknown table
            if not schema.has_column(canonical, name):
                errors.append(f"Column '{name}' does not exist in table {canonical}. "
                              f"Its columns are: {', '.join(schema.tables[canonical].column_names())}.")
        else:
            if virtual or name.lower() in select_aliases:
                continue  # may come from a CTE/subquery or be a select-list alias; not a schema column
            if real_tables and not any(schema.has_column(t, name) for t in real_tables):
                errors.append(f"Column '{name}' does not exist in any referenced table ({', '.join(real_tables)}).")
    return errors


def validate(sql: str, schema: Optional[Schema] = None) -> Tuple[exp.Expression, List[str]]:
    """Parse, enforce read-only, and (if a schema is given) check identifiers. Raises SQLGuardError for hard failures."""
    expr = parse(sql)
    assert_read_only(expr)
    errors = validate_against_schema(expr, schema) if schema else []
    return expr, errors
