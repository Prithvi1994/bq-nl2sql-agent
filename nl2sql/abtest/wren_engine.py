"""Wren-backed planning and execution for the experiment path.

This is the single place SQL touches the semantic layer. The agent and the tools ask
for a query by intent; `plan` hands it to Wren, which expands MDL models and measures
into concrete BigQuery, and `run` executes the result against the offline mirror.

Wren is called in-process via `wren.mdl.transform_sql` (an 11 ms round trip through
`wren-core`, which is DataFusion) rather than through the `wren` CLI, because a
subprocess per question costs more than the plan itself.

The one deviation from production: the compiled MDL targets BigQuery and emits fully
qualified names like `analytics.ab.fact_assigned_with_pre`. The offline mirror keeps
those as bare table names, so `run` strips the catalog qualification before handing
the SQL to DuckDB. Against real BigQuery that rewrite is skipped — it exists only so
the same query text can be verified offline, and it is recorded in the query log.
"""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# wren_engine.py -> abtest -> nl2sql -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROJECT = REPO_ROOT / "wren_project"


class WrenError(RuntimeError):
    """Raised when the semantic layer cannot plan the query. Never a fallback."""


@dataclass
class PlannedQuery:
    """What Wren produced, plus how it got there."""

    sql: str                      # the SQL the caller asked for, by intent
    planned_sql: str              # what Wren expanded it to
    dialect: str
    plan_ms: int
    models: List[str] = field(default_factory=list)
    measures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "planned_sql": self.planned_sql,
            "dialect": self.dialect,
            "plan_ms": self.plan_ms,
            "models": self.models,
            "measures": self.measures,
        }


# The compiled manifest is read once and cached; building it costs ~100 ms and the
# bytes never change within a process unless someone edits the YAML mid-run.
_MANIFEST_CACHE: Dict[str, str] = {}


def load_manifest(project: Path | str = DEFAULT_PROJECT) -> str:
    """Compiled MDL as the base64 string `wren-core` expects."""
    key = str(Path(project).resolve())
    if key not in _MANIFEST_CACHE:
        mdl_path = Path(project) / "target" / "mdl.json"
        if not mdl_path.exists():
            raise WrenError(
                f"{mdl_path} not found. Run: .venv/bin/wren context build --path {project}"
            )
        _MANIFEST_CACHE[key] = base64.b64encode(mdl_path.read_bytes()).decode("ascii")
    return _MANIFEST_CACHE[key]


def invalidate_cache(project: Path | str = DEFAULT_PROJECT) -> None:
    _MANIFEST_CACHE.pop(str(Path(project).resolve()), None)


_MEASURE_NAMES = (
    "purchase_rate", "add_to_cart_rate", "revenue_per_user", "revenue_per_purchaser",
    "sessions_per_user", "pageviews_per_user", "days_active", "session_duration_s",
    "assigned_users", "active_users", "purchasing_users", "add_to_cart_users",
    "total_revenue", "total_purchases", "total_sessions", "total_pageviews",
)
_MODEL_NAMES = ("experiment_user", "experiment_daily", "assigned_population")


def plan(
    sql: str,
    project: Path | str = DEFAULT_PROJECT,
    dialect: str = "bigquery",
) -> PlannedQuery:
    """Expand `sql` through the MDL. Fails closed if Wren cannot plan it."""
    try:
        from wren.mdl import transform_sql
    except ImportError as exc:  # pragma: no cover
        raise WrenError(
            "wrenai is not installed. Run: uv pip install wrenai"
        ) from exc

    manifest = load_manifest(project)
    started = time.perf_counter()
    try:
        # data_source must be the *dialect* string. Omitting it makes Wren plan with
        # DataFusion's default function set, where IF() and SAFE_DIVIDE() are unknown
        # and every metric expression fails.
        planned = transform_sql(manifest, sql, data_source=dialect)
    except Exception as exc:  # noqa: BLE001 - wren-core raises its own error type
        raise WrenError(f"Wren could not plan this query: {exc}") from exc
    plan_ms = int((time.perf_counter() - started) * 1000)

    return PlannedQuery(
        sql=sql,
        planned_sql=planned,
        dialect=dialect,
        plan_ms=plan_ms,
        models=[m for m in _MODEL_NAMES if re.search(rf"\b{m}\b", sql)],
        measures=[m for m in _MEASURE_NAMES if re.search(rf"\b{m}\b", sql)],
    )


def to_mirror_sql(planned_sql: str) -> str:
    """Strip BigQuery catalog qualification so the offline mirror can resolve it.

    Production BigQuery needs `analytics.ab.fact_assigned_with_pre`; the DuckDB mirror
    registered the bare name. This is the only dialect accommodation in the path, and
    it is applied after planning so the planned SQL stays BigQuery for the log.
    """
    return re.sub(r"\b(?:analytics|ab)\.", "", planned_sql)


def run(
    sql: str,
    db_path: str,
    project: Path | str = DEFAULT_PROJECT,
    row_limit: int = 1_000_000,
) -> Dict[str, Any]:
    """Plan through Wren, then execute against the offline mirror.

    Returns the query log entry plus rows. Raises rather than returning a partial
    result: a silent empty result reads as "no data" and is indistinguishable from a
    wrong answer.
    """
    from ..bq_exec import run_bq_query

    planned = plan(sql, project=project)
    exec_sql = to_mirror_sql(planned.planned_sql)
    res = run_bq_query(exec_sql, db_path, row_limit=row_limit)
    if not res.ok:
        raise WrenError(f"planned query failed to execute: {res.error}")
    return {
        "rows": res.rows,
        "columns": res.columns,
        "plan": planned.to_dict(),
        "truncated": res.truncated,
        "row_count": len(res.rows),
    }


def metric_card() -> str:
    """The catalog, rendered for a prompt. Read from the MDL, never hard-coded."""
    import yaml  # local: only the prompt path needs it

    project = Path(DEFAULT_PROJECT)
    card: List[str] = ["Metric catalog. Use these names in the semantic layer:"]
    for cube in sorted((project / "cubes").glob("*/metadata.yml")):
        spec = yaml.safe_load(cube.read_text(encoding="utf-8")) or {}
        root = spec.get("base_object", "?")
        card.append(f"\nCube {spec.get('name')} (base: {root}):")
        for measure in spec.get("measures", []) or []:
            desc = " ".join((measure.get("description") or "").split())
            card.append(f"  - {measure['name']}: {desc}")
    return "\n".join(card)


def population_rules() -> str:
    """Population policies. These are the rules an MDL cannot express."""
    path = Path(DEFAULT_PROJECT) / "knowledge" / "rules" / "population.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""
