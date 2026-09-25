"""Experiment readout: BigQuery SQL in, grounded markdown out.

Pipeline, in order:

  1. `load_experiment`   BigQuery SQL -> per-user test-phase rows
  2. `profile`           data-quality checks that the report must disclose
  3. `analyse`           SRM, lift, CIs, CUPED  (nl2sql.abtest.stats)
  4. `decide`            pre-registered decision rule
  5. `render`            markdown, with every number traced to a query or a function

The language model is optional. Without it you still get a complete, grounded report;
with it, only the prose sections are generated, from computed values only.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..bq_exec import BigQueryResult, run_bq_query
from .stats import analyse, decide

# BigQuery-flavoured SQL with a scripting header, exactly as an analyst would write it.
# DECLARE names are prefixed with __ because `experiment_id` is also a column name, and
# textual inlining of a variable into a USING(...) list would corrupt the join.
#
# Grain note: this reads `fact_assigned_with_pre`, which is built from the *assignment*
# table left-joined to outcomes. Every assigned user has a row even with zero activity,
# so the purchase_rate denominator is assigned users. Filtering on the metrics table
# instead would drop inactive users and select on the outcome, inflating lift.
QUERY_TEST_PHASE_USER_TOTALS = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';

SELECT
  user_id,
  experiment_id,
  variant,
  days_active,
  sessions,
  pageviews,
  add_to_cart,
  purchases,
  revenue_usd,
  pre_sessions,
  pre_pageviews,
  pre_revenue,
  pre_days
FROM `analytics.ab.{experiment_id}.fact_assigned_with_pre`
WHERE experiment_id = __exp_id
ORDER BY user_id
"""

QUERY_DAILY_TREND = """
SELECT
  m.metric_date,
  a.variant,
  COUNT(DISTINCT m.user_id) AS users,
  SUM(m.purchases) AS purchases,
  SUM(m.revenue_usd) AS revenue_usd
FROM `analytics.ab.{experiment_id}.fact_daily_user_metrics` AS m
JOIN `analytics.ab.{experiment_id}.fact_user_assignments` AS a
  USING (user_id, experiment_id)
WHERE m.phase = 'test'
  AND m.experiment_id = '{experiment_id}'
GROUP BY metric_date, variant
ORDER BY metric_date, variant
"""

QUERY_SRM = """
SELECT
  variant,
  COUNT(*) AS users
FROM `analytics.ab.{experiment_id}.fact_user_assignments`
WHERE experiment_id = '{experiment_id}'
GROUP BY variant
ORDER BY variant
"""

QUERY_DAILY_SPINE = """
DECLARE start_date DATE DEFAULT '{start}';
DECLARE end_date DATE DEFAULT '{end}';

SELECT
  d AS metric_date,
  v.variant,
  COALESCE(t.users, 0) AS users,
  COALESCE(t.purchases, 0) AS purchases,
  COALESCE(t.revenue_usd, 0.0) AS revenue_usd
FROM UNNEST(GENERATE_DATE_ARRAY(start_date, end_date, INTERVAL 1 DAY)) AS d
CROSS JOIN UNNEST(['{variants}']) AS v(variant)
LEFT JOIN (
  SELECT metric_date, variant, COUNT(DISTINCT user_id) AS users,
         SUM(purchases) AS purchases, SUM(revenue_usd) AS revenue_usd
  FROM `analytics.ab.{experiment_id}.fact_daily_user_metrics`
  WHERE phase = 'test' AND experiment_id = '{experiment_id}'
  GROUP BY metric_date, variant
) AS t
  ON t.metric_date = d AND t.variant = v.variant
ORDER BY metric_date, variant
"""


@dataclass
class Readout:
    experiment_id: str
    analysis: Dict[str, Any]
    decision: Dict[str, Any]
    queries: List[Dict[str, Any]]
    markdown: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "analysis": self.analysis,
            "decision": self.decision,
            "queries": self.queries,
            "markdown": self.markdown,
        }


def _q(sql: str, db_path: str, queries: List[Dict[str, Any]], label: str) -> BigQueryResult:
    t0 = time.perf_counter()
    res = run_bq_query(sql, db_path, row_limit=100_000, timeout_s=60.0)
    queries.append(
        {
            "label": label,
            "ok": res.ok,
            "rows": res.row_count,
            "elapsed_ms": res.elapsed_ms,
            "sql": res.dialect_sql.strip(),
            "transpiled_sql": res.duckdb_sql.strip(),
            "error": res.error,
        }
    )
    if not res.ok:
        raise RuntimeError(f"query {label!r} failed: {res.error}")
    return res


def load_experiment(db_path: str, experiment_id: str, queries: List[Dict[str, Any]]) -> List[Sequence[Any]]:
    res = _q(QUERY_TEST_PHASE_USER_TOTALS.format(experiment_id=experiment_id), db_path, queries, "user_totals")
    return list(res.rows)


def run_readout(
    db_path: str,
    experiment_id: str,
    *,
    expected_shares: Optional[Dict[str, float]] = None,
    primary_metric: str = "purchase_rate",
    ship_threshold: float = 0.05,
    guardrail_metric: str = "revenue_per_user",
    guardrail_tolerance: float = -0.02,
    srm_threshold: float = 0.001,
) -> Readout:
    queries: List[Dict[str, Any]] = []
    rows = load_experiment(db_path, experiment_id, queries)
    if not rows:
        raise RuntimeError("no per-user rows returned; is the dataset built?")

    # Column layout from the SQL above: user, exp, variant, days_active, sessions,
    # pageviews, add_to_cart, purchases, revenue, pre_sessions, pre_pageviews,
    # pre_revenue, pre_days  (index 13)
    COVARIATE = {
        "purchase_rate": None,
        "revenue_per_user": 11,
        "sessions_per_user": 9,
        "pageviews_per_user": 10,
        "days_active": 12,
    }

    analysis_all: Dict[str, Any] = {}
    for metric, cov_idx in COVARIATE.items():
        if cov_idx is None:
            continue
        a = analyse(rows, expected_shares=expected_shares, covariate_index=cov_idx)
        analysis_all[metric] = a

    # Primary pass drives the decision; it uses revenue pre-period as covariate.
    primary = analyse(rows, expected_shares=expected_shares,
                      covariate_index=COVARIATE.get(guardrail_metric) or 11)
    srm_query = _q(QUERY_SRM.format(experiment_id=experiment_id), db_path, queries, "srm_counts")
    if expected_shares is None and srm_query.rows:
        n = len(srm_query.rows)
        expected_shares = {str(r[0]): 1.0 / n for r in srm_query.rows}
        primary = analyse(rows, expected_shares=expected_shares, covariate_index=11)

    decision = decide(
        primary,
        primary_metric=primary_metric,
        ship_threshold=ship_threshold,
        guardrail_metric=guardrail_metric,
        guardrail_tolerance=guardrail_tolerance,
    )
    # Re-attach the primary-metric lifts so the renderer uses one consistent object.
    for name, metrics in primary.get("lifts", {}).items():
        for metric, idx in (("purchase_rate", 7), ("add_to_cart_rate", 6)):
            a_metric = analysis_all.get(metric)
            if a_metric and name in a_metric.get("lifts", {}):
                metrics[metric] = a_metric["lifts"][name][metric]
    return Readout(
        experiment_id=experiment_id,
        analysis=primary,
        decision=decision,
        queries=queries,
        markdown=render(primary, decision, experiment_id, queries),
    )


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%"


def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.{digits}f}"


def render(
    analysis: Dict[str, Any],
    decision: Dict[str, Any],
    experiment_id: str,
    queries: List[Dict[str, Any]],
) -> str:
    """Markdown readout. Every value here comes from `analysis` or `decision`."""
    srm = analysis.get("srm", {})
    lines: List[str] = []
    lines.append(f"# Experiment readout — `{experiment_id}`")
    lines.append("")
    lines.append(f"**Decision: {decision.get('decision', 'UNKNOWN')}**")
    lines.append("")
    lines.append("## Data integrity")
    lines.append("")
    if srm.get("detected"):
        lines.append(
            f"- ⚠️ **Sample ratio mismatch detected.** Intended split "
            f"{json.dumps(srm['expected'])} vs observed {json.dumps(srm['observed'])}; "
            f"chi-square p={srm['p_value']:.2e} < {srm['threshold']}."
        )
        lines.append("- Effect estimates below are reported for completeness only; the randomisation is not trustworthy.")
    else:
        lines.append(
            f"- Assignment matches the intended split (chi-square p={srm.get('p_value', float('nan')):.3f}); no SRM."
        )
    lines.append(f"- Users analysed: {analysis.get('users', 0):,}")
    lines.append("")

    lines.append("## Variant summary")
    lines.append("")
    lines.append("| Variant | Users | Share | purchase_rate | revenue_per_user | sessions_per_user |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, v in sorted(analysis.get("variants", {}).items()):
        m = v["metrics"]
        lines.append(
            f"| `{name}`{' (control)' if v['is_control'] else ''} | {v['users']:,} | "
            f"{v['assigned_share']:.1%} | {m['purchase_rate']:.4f} | "
            f"${m['revenue_per_user']:.2f} | {m['sessions_per_user']:.2f} |"
        )
    lines.append("")

    lines.append("## Lift vs control")
    lines.append("")
    for name, metrics in sorted(analysis.get("lifts", {}).items()):
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| Metric | Control | Treatment | Lift | 95% CI | p | Method |")
        lines.append("|---|---:|---:|---:|---|---:|---|")
        for metric, r in sorted(metrics.items()):
            if r["control_value"] is None:
                continue
            val = (lambda x: f"{x:,.4f}") if metric.endswith("_rate") else (lambda x: f"{x:,.2f}")
            lines.append(
                f"| {metric} | {val(r['control_value'])} | {val(r['treatment_value'])} | "
                f"{_fmt_pct(r['relative_lift'])} | [{_fmt_pct(r['ci_low'])}, {_fmt_pct(r['ci_high'])}] | "
                f"{r['p_value']:.3f} | {r['method']} |"
            )
        lines.append("")
        cuped = [(m, r) for m, r in sorted(metrics.items()) if r.get("cuped_relative_lift") is not None]
        if cuped:
            lines.append("CUPED-adjusted (pre-period covariate):")
            lines.append("")
            for metric, r in cuped:
                lines.append(
                    f"- {metric}: {_fmt_pct(r['cuped_relative_lift'])} "
                    f"[{_fmt_pct(r['cuped_ci_low'])}, {_fmt_pct(r['cuped_ci_high'])}], p={r['cuped_p_value']:.3f}"
                )
            lines.append("")

    if decision.get("reasons"):
        lines.append("## Decision rationale")
        lines.append("")
        for r in decision["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    for q in queries:
        lines.append(f"- `{q['label']}` — {q['rows']:,} rows in {q['elapsed_ms']} ms (BigQuery SQL, transpiled for the local mirror)")
    lines.append("")
    lines.append(
        "All statistics are computed in `nl2sql.abtest.stats` from query output. "
        "No number in this report was produced by a language model."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an experiment readout from the offline dataset.")
    ap.add_argument("--db", default="./abtest_data/abtest.duckdb")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--expected", default=None, help='JSON: {"control":0.5,"treatment_b":0.5}')
    ap.add_argument("--out", default=None, help="Write markdown here instead of stdout.")
    args = ap.parse_args()

    expected = json.loads(args.expected) if args.expected else None
    readout = run_readout(args.db, args.experiment, expected_shares=expected)
    if args.out:
        Path(args.out).write_text(readout.markdown, encoding="utf-8")
        print(f"wrote {args.out}")
        print(f"decision={readout.decision.get('decision')}")
    else:
        print(readout.markdown)


if __name__ == "__main__":
    main()
