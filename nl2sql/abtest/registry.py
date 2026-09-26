"""The experiment registry: domain → experiment → physical table.

This is the platform piece. Agentic tools need to answer two questions before any
SQL exists:

  1. which experiments exist in a domain?            (discover)
  2. given domain + experiment id, which table?      (resolve)

The registry is one DuckDB table, `dim_experiment_registry`, written by the
platform when an experiment lands. Physical table resolution is a convention,
not a lookup the LLM can get wrong: `exp_<experiment_id>` in a fixed schema.

Design rules
------------
- One row per experiment. `domain` is the routing key.
- `table_name` is the physical table the experiment's tracker landed. It is
  recorded at onboard time; agents resolve THROUGH the registry, never by
  guessing the name from the id.
- `metric_columns` lists the metric columns the table actually carries (the
  shared contract plus any domain extras). Tooling reads this list to build the
  query, so a domain column (e.g. `plan_tier`) becomes metric/dimension without
  new code.
- `daily` is fixed True for now: the contract is one row per day per variant.
"""
from __future__ import annotations

import re

import duckdb
from typing import Any, Dict, List, Optional, Sequence

from ..bq_exec import run_bq_query


class RegistryError(ValueError):
    """Fail-closed: an experiment or domain that does not exist is a refusal."""


REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS dim_experiment_registry (
    experiment_id   VARCHAR PRIMARY KEY,
    domain          VARCHAR NOT NULL,
    name            VARCHAR,
    table_name      VARCHAR NOT NULL,
    schema          VARCHAR DEFAULT 'exp',
    started_on      DATE,
    ended_on        DATE,
    variants        VARCHAR,          -- 'control,treatment_a,treatment_b'
    metric_columns  VARCHAR,          -- 'users,sessions,pageviews,purchases,revenue_usd'
    dimension_columns VARCHAR,        -- 'country,platform'
    north_star_metric VARCHAR,        -- the headline KPI; narration reads verdicts on it
    guardrail_metric VARCHAR,
    ship_threshold  DOUBLE,
    guardrail_tolerance DOUBLE
)
"""


def ensure_registry(db_path: str) -> None:
    """CREATE the registry table if absent. Direct rw connection: the read-only
    query executor refuses DDL/DML by design, so infra writes never share the
    agent's query surface."""
    con = duckdb.connect(db_path)
    try:
        con.execute(REGISTRY_DDL)
    finally:
        con.close()


def register(
    db_path: str,
    experiment_id: str,
    domain: str,
    *,
    name: str = "",
    table_name: Optional[str] = None,
    schema: str = "exp",
    started_on: str = "",
    ended_on: str = "",
    variants: Sequence[str] = (),
    metric_columns: Sequence[str] = (),
    dimension_columns: Sequence[str] = (),
    north_star_metric: str = "",
    guardrail_metric: str = "",
    ship_threshold: Optional[float] = None,
    guardrail_tolerance: Optional[float] = None,
) -> None:
    """Insert or update one experiment. Idempotent: re-registering replaces."""
    ensure_registry(db_path)
    table = table_name or f"exp_{experiment_id}"
    # Ids and domains are identifiers, never free text; reject anything that could
    # become a second SQL surface.
    if not re.fullmatch(r"[a-z0-9_]+", experiment_id):
        raise RegistryError(f"invalid experiment id {experiment_id!r}")
    if not re.fullmatch(r"[a-z0-9_]+", domain):
        raise RegistryError(f"invalid domain {domain!r}")
    # Idempotent replace: one row per experiment. Writes go through a direct
    # rw connection — the read-only query executor refuses DML by design.
    identifier = experiment_id  # validated [a-z0-9_] above
    con = duckdb.connect(db_path)
    try:
        con.execute(
            "DELETE FROM dim_experiment_registry WHERE experiment_id = ?",
            [identifier],
        )
        con.execute(
            "INSERT INTO dim_experiment_registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                identifier, domain, name or experiment_id, table, schema,
                started_on or None, ended_on or None,
                ",".join(variants), ",".join(metric_columns),
                ",".join(dimension_columns), north_star_metric, guardrail_metric,
                ship_threshold, guardrail_tolerance,
            ],
        )
    finally:
        con.close()


def _rows(db_path: str) -> List[Dict[str, Any]]:
    ensure_registry(db_path)
    res = run_bq_query(
        "SELECT experiment_id, domain, name, table_name, schema, started_on, ended_on, "
        "variants, metric_columns, dimension_columns, north_star_metric, guardrail_metric, "
        "ship_threshold, guardrail_tolerance FROM dim_experiment_registry",
        db_path, row_limit=10_000,
    )
    if not res.ok:
        raise RegistryError(f"registry unreadable: {res.error}")
    cols = ["experiment_id", "domain", "name", "table_name", "schema",
            "started_on", "ended_on", "variants", "metric_columns",
            "dimension_columns", "north_star_metric", "guardrail_metric",
            "ship_threshold", "guardrail_tolerance"]
    return [dict(zip(cols, r)) for r in res.rows]


def experiments_for_domain(db_path: str, domain: str) -> List[Dict[str, Any]]:
    """Tool 1: all experiments in a domain. Unknown domain is fail-closed."""
    rows = _rows(db_path)
    hits = [r for r in rows if r["domain"] == domain]
    if not hits:
        known = sorted({r["domain"] for r in rows})
        raise RegistryError(f"no experiments in domain {domain!r}. Known: {known}")
    return hits


def resolve(db_path: str, domain: str, experiment_id: str) -> Dict[str, Any]:
    """Tool 2: the experiment record for domain+id, INCLUDING its physical table
    and the columns it actually carries. This is the only sanctioned way to get
    a table name; guessing exp_<id> from a question is not.

    Independently verify the table actually exists before returning it: a
    registry row pointing at a dropped table is exactly the kind of
    plausible-but-wrong pointer that reads as truth downstream.
    """
    rows = _rows(db_path)
    hit = next((r for r in rows
                if r["domain"] == domain and r["experiment_id"] == experiment_id), None)
    if hit is None:
        in_domain = [r["experiment_id"] for r in rows if r["domain"] == domain]
        raise RegistryError(
            f"experiment {experiment_id!r} does not exist in domain {domain!r}. "
            + (f"Experiments in {domain}: {in_domain}" if in_domain else
               f"Known domains: {sorted({r['domain'] for r in rows})}")
        )
    # existence check runs through the same read-only executor the agent uses;
    # duckdb_tables() is consulted directly because information_schema does not
    # see cross-schema objects once _unqualify_tables has stripped qualification.
    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?",
            [hit["table_name"]],
        ).fetchone()[0] > 0
    finally:
        con.close()
    return {**hit, "table_exists": exists}


def domains(db_path: str) -> List[str]:
    return sorted({r["domain"] for r in _rows(db_path)})


def load(db_path: str) -> List[Dict[str, Any]]:
    return _rows(db_path)
