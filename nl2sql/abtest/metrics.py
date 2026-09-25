"""The metric catalog: the single source of truth for what a metric means.

A business user asks "which variant had the highest revenue lift?". That question has
to become BigQuery SQL, and the SQL has to mean the same thing every time. Free-text
generation cannot guarantee that, so the definitions live here as code and the model
picks a definition rather than inventing one.

Each MetricDef carries:
  * name, description, and the keywords a user might say instead
  * the SQL expression that computes it, per unit (user / session / day)
  * the aggregation to use when aggregating across users
  * the direction that counts as an improvement

Adding a metric is one entry here. Nothing else changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .stats import EPS, Z95, _binary_lift, _continuous_lift, _cuped
import numpy as np
from scipy import stats


@dataclass
class MetricDef:
    name: str
    unit: str                      # "user" | "event" | "day"
    description: str
    sql: Dict[str, str]            # unit -> SQL expression
    agg: str                       # how to roll the unit up to a per-user value
    keywords: List[str] = field(default_factory=list)
    higher_is_better: bool = True
    is_binary: bool = False
    category: str = "engagement"
    population: str = "all"
    """Who the metric is measured over.

    "all" = every assigned user, including those with zero activity. That is the
    correct denominator for conversion rate and revenue per user.
    "purchasers" = only users with at least one purchase. Averaging AOV over all
    users silently turns it into revenue per user, because non-purchasers contribute
    zero revenue and a zero denominator. AOV must be measured over buyers.
    """

    def expression(self, unit: Optional[str] = None) -> str:
        u = unit or self.unit
        if u not in self.sql:
            raise KeyError(f"metric {self.name!r} has no SQL for unit {u!r}")
        return self.sql[u]

    def window_expression(self) -> str:
        """The metric summed over the joined window rows, per user.

        Used in the one query that joins assigned_population to experiment_window.
        The join produces one row per user-day, so the per-user total is SUM over
        those rows. Binary metrics sum their 0/1 indicator, which is how a
        "purchased at least once" flag is recovered from a daily grain.
        """
        col = {
            "purchase_rate": "CASE WHEN w.purchases > 0 THEN 1.0 ELSE 0.0 END",
            "add_to_cart_rate": "CASE WHEN w.add_to_cart > 0 THEN 1.0 ELSE 0.0 END",
            "revenue_per_user": "w.revenue_usd",
            "revenue_per_purchaser": "w.revenue_usd",
            "sessions_per_user": "w.sessions",
            "pageviews_per_user": "w.pageviews",
            "days_active": "1.0",
            "session_duration_s": "w.session_duration_s",
        }.get(self.name)
        if col is None:
            raise KeyError(
                f"no window expression for metric {self.name!r}; add it so a windowed "
                "lift cannot be silently computed over the wrong column"
            )
        return f"SUM({col})"

    def daily_expression(self) -> str:
        """The metric over ONE user-day row of experiment_daily.

        This is a per-active-user value for a single day, not a rate over a
        population, so it must not divide by any user count: experiment_daily only
        has a row for users who were active that day, so "users" here is the active
        population and the caller is asking for a daily trend, not a rate.
        """
        col = {
            "purchase_rate": "CASE WHEN purchases > 0 THEN 1.0 ELSE 0.0 END",
            "add_to_cart_rate": "CASE WHEN add_to_cart > 0 THEN 1.0 ELSE 0.0 END",
            "revenue_per_user": "revenue_usd",
            "revenue_per_purchaser": (
                "CASE WHEN purchases > 0 THEN revenue_usd / purchases ELSE NULL END"
            ),
            "sessions_per_user": "sessions",
            "pageviews_per_user": "pageviews",
            "days_active": "1.0",
            "session_duration_s": "session_duration_s",
        }.get(self.name)
        if col is None:
            raise KeyError(
                f"no daily expression for metric {self.name!r}; add it so a trend "
                "cannot be silently computed over the wrong population"
            )
        return f"AVG({col})"

    def segment_expression(self) -> str:
        """How to roll the metric up over a segment of assigned users.

        experiment_user holds one row per assignment with metrics already
        pre-aggregated, so a segment breakdown is just AVG of the raw column. It
        must NOT be AVG of `expression("user")`: that expression is itself an
        aggregate (SUM or MAX) because it is meant to be evaluated per user, and
        nesting an aggregate inside AVG fails to plan.

        The measure that the segment rolls up is therefore chosen here, by metric
        shape, rather than derived from the per-user expression:

          * binary metrics   -> AVG of the 0/1 indicator
          * "sum" metrics    -> AVG of the raw column (already per-user total)
          * "max" metrics    -> AVG of the raw column (already per-user count)

        For a sum-metric the per-user total and the segment mean are the same
        number averaged over users, which is what "revenue per user by country"
        means. For AOV the population filter (`purchasers`) still applies, so the
        average is taken only over users who bought.
        """
        col = {
            "purchase_rate": "CASE WHEN purchases > 0 THEN 1.0 ELSE 0.0 END",
            "add_to_cart_rate": "CASE WHEN add_to_cart > 0 THEN 1.0 ELSE 0.0 END",
            "revenue_per_user": "revenue_usd",
            "revenue_per_purchaser": (
                "CASE WHEN purchases > 0 THEN revenue_usd / purchases ELSE NULL END"
            ),
            "sessions_per_user": "sessions",
            "pageviews_per_user": "pageviews",
            "days_active": "days_active",
            "session_duration_s": "session_duration_s",
        }.get(self.name)
        if col is None:
            raise KeyError(
                f"no segment expression for metric {self.name!r}; add it to "
                "MetricDef.segment_expression so a segment breakdown cannot be "
                "silently wrong"
            )
        return f"AVG({col})"

    def per_user(self, rows: Sequence[Sequence[Any]], col: int) -> np.ndarray:
        """Reduce a per-row column to a per-user value using this metric's aggregation."""
        vals = np.asarray(
            [0.0 if r[col] is None else float(r[col]) for r in rows], dtype=float
        )
        if self.is_binary:
            return (vals > 0).astype(float)
        if self.agg == "sum":
            return vals
        if self.agg == "mean":
            return vals
        return vals


METRICS: List[MetricDef] = [
    MetricDef(
        name="purchase_rate",
        unit="user",
        description="Share of assigned users who made at least one purchase during the window.",
        sql={
            "user": "MAX(CASE WHEN purchases > 0 THEN 1 ELSE 0 END)",
            "pre": "CASE WHEN p.pre_purchases > 0 THEN 1 ELSE 0 END",
            "event": "CASE WHEN purchases > 0 THEN 1 ELSE 0 END",
            "day": "CASE WHEN SUM(purchases) > 0 THEN 1 ELSE 0 END",
        },
        agg="max",
        keywords=["conversion", "purchase rate", "convert", "buyers", "purchased",
                  "conversion rate", "did they buy", "orders", "purchase"],
        is_binary=True,
        category="conversion",
    ),
    MetricDef(
        name="add_to_cart_rate",
        unit="user",
        description="Share of assigned users who added something to the cart.",
        sql={
            "user": "MAX(CASE WHEN add_to_cart > 0 THEN 1 ELSE 0 END)",
            "pre": "CASE WHEN p.pre_add_to_cart > 0 THEN 1 ELSE 0 END",
            "event": "CASE WHEN add_to_cart > 0 THEN 1 ELSE 0 END",
            "day": "CASE WHEN SUM(add_to_cart) > 0 THEN 1 ELSE 0 END",
        },
        agg="max",
        keywords=["cart", "add to cart", "basket", "cart adds"],
        is_binary=True,
        category="conversion",
    ),
    MetricDef(
        name="revenue_per_user",
        unit="user",
        description="Total revenue in USD divided by assigned users, including users who spent nothing.",
        sql={
            "user": "SUM(revenue_usd)",
            "pre": "p.pre_revenue",
            "event": "revenue_usd",
            "day": "SUM(revenue_usd)",
        },
        agg="sum",
        keywords=["revenue", "sales", "gmv", "dollars", "income", "booking", "arpu",
                  "revenue per user", "average order", "aov", "spend"],
        category="revenue",
    ),
    MetricDef(
        name="revenue_per_purchaser",
        unit="user",
        description="Average order value: revenue divided by the number of users who purchased.",
        sql={
            "user": "CASE WHEN SUM(purchases) > 0 THEN SUM(revenue_usd) / SUM(purchases) ELSE NULL END",
            "pre": "CASE WHEN p.pre_purchases > 0 THEN p.pre_revenue / p.pre_purchases ELSE NULL END",
            "event": "revenue_usd",
            "day": "CASE WHEN SUM(purchases) > 0 THEN SUM(revenue_usd) / SUM(purchases) ELSE NULL END",
        },
        agg="mean",
        keywords=["aov", "average order value", "order value", "basket size", "ticket"],
        category="revenue",
        population="purchasers",
    ),
    MetricDef(
        name="sessions_per_user",
        unit="user",
        description="Sessions per assigned user, including users with no sessions.",
        sql={"user": "SUM(sessions)",
            "pre": "p.pre_sessions", "event": "sessions", "day": "SUM(sessions)"},
        agg="sum",
        keywords=["sessions", "engagement", "visits", "frequency", "activity",
                  "sessions per user"],
        category="engagement",
    ),
    MetricDef(
        name="pageviews_per_user",
        unit="user",
        description="Pageviews per assigned user.",
        sql={"user": "SUM(pageviews)",
            "pre": "p.pre_pageviews", "event": "pageviews", "day": "SUM(pageviews)"},
        agg="sum",
        keywords=["pageviews", "views", "browsing", "pages"],
        category="engagement",
    ),
    MetricDef(
        name="days_active",
        unit="user",
        description="Distinct days the user was active in the window.",
        sql={"user": "COUNT(DISTINCT metric_date)",
            "pre": "p.pre_days", "event": "1", "day": "1"},
        agg="max",
        keywords=["days active", "retention", "active days", "frequency",
                  "engagement days", "dau"],
        category="engagement",
    ),
    MetricDef(
        name="session_duration_s",
        unit="user",
        description="Total session duration in seconds per assigned user.",
        sql={"user": "SUM(session_duration_s)",
            "pre": "p.pre_session_duration_s", "event": "session_duration_s",
             "day": "SUM(session_duration_s)"},
        agg="sum",
        keywords=["duration", "time on site", "session length", "engagement time", "seconds"],
        category="engagement",
    ),
]

METRIC_BY_NAME: Dict[str, MetricDef] = {m.name: m for m in METRICS}


def card() -> str:
    """Compact catalog for the prompt. Names, units and SQL, no prose bloat."""
    lines = ["Metric catalog (use these names and expressions verbatim):"]
    for m in METRICS:
        lines.append(f"  {m.name} [{m.category}] ({m.unit}-grain): {m.description}")
        lines.append(f"      per-user SQL: {m.expression('user')}")
    return "\n".join(lines)


def find_metric(text: str) -> Optional[MetricDef]:
    """Resolve free text to a metric by exact name, then by keyword.

    Keyword match is longest-first so "revenue per user" beats "revenue". Returns
    None when nothing matches: an unresolved metric must fail, not default.
    """
    t = text.lower().strip()
    if t in METRIC_BY_NAME:
        return METRIC_BY_NAME[t]
    for m in sorted(METRICS, key=lambda x: -len(x.name)):
        if m.name in t:
            return m
    best: Optional[MetricDef] = None
    best_len = 0
    for m in METRICS:
        for kw in m.keywords:
            if kw in t and len(kw) > best_len:
                best, best_len = m, len(kw)
    return best


def all_matches(text: str) -> List[MetricDef]:
    """Every metric mentioned, in order of appearance. Used for guardrail detection."""
    t = text.lower()
    hits: List[MetricDef] = []
    for m in METRICS:
        if m.name in t or any(kw in t for kw in m.keywords):
            hits.append(m)
    return hits


# ---------------------------------------------------------------------------
# Canonical BigQuery SQL. These are the queries the agent should emit; they are
# real BQ SQL with DECLARE scripting, tested against the offline mirror.
# ---------------------------------------------------------------------------

BQ_SQL_VARIANT_METRICS = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';
DECLARE __start DATE DEFAULT DATE('{start}');
DECLARE __end DATE DEFAULT DATE('{end}');

WITH scoped AS (
  SELECT
    m.user_id,
    m.variant,
    {expr} AS metric_value
  FROM `analytics.ab.{experiment_id}.fact_daily_assigned` AS m
  WHERE m.experiment_id = __exp_id
    AND m.phase = 'test'
    AND m.metric_date BETWEEN __start AND __end
  GROUP BY m.user_id, m.variant
)
SELECT
  variant,
  COUNT(*) AS assigned_users,
  SUM(metric_value) AS total_value,
  SAFE_DIVIDE(SUM(metric_value), COUNT(*)) AS metric_per_user
FROM scoped
GROUP BY variant
ORDER BY variant
"""

BQ_SQL_DAILY_TREND = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';
DECLARE __start DATE DEFAULT DATE('{start}');
DECLARE __end DATE DEFAULT DATE('{end}');

SELECT
  metric_date,
  variant,
  COUNT(DISTINCT user_id) AS active_users,
  {expr} AS metric_value
FROM `analytics.ab.{experiment_id}.fact_daily_assigned`
WHERE experiment_id = __exp_id
  AND phase = 'test'
  AND metric_date BETWEEN __start AND __end
GROUP BY metric_date, variant
ORDER BY metric_date, variant
"""

BQ_SQL_CUMULATIVE = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';
DECLARE __start DATE DEFAULT DATE('{start}');
DECLARE __end DATE DEFAULT DATE('{end}');

WITH daily AS (
  SELECT
    metric_date,
    variant,
    {expr} AS daily_value
  FROM `analytics.ab.{experiment_id}.fact_daily_assigned`
  WHERE experiment_id = __exp_id
    AND phase = 'test'
    AND metric_date BETWEEN __start AND __end
  GROUP BY metric_date, variant
)
SELECT
  metric_date,
  variant,
  SUM(daily_value) OVER (
    PARTITION BY variant ORDER BY metric_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS cumulative_value
FROM daily
ORDER BY metric_date, variant
"""

BQ_SQL_SEGMENT = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';
DECLARE __start DATE DEFAULT DATE('{start}');
DECLARE __end DATE DEFAULT DATE('{end}');
DECLARE __dimension STRING DEFAULT '{dimension}';

WITH scoped AS (
  SELECT
    m.user_id,
    m.variant,
    m.{dimension} AS segment_value,
    {expr} AS metric_value
  FROM `analytics.ab.{experiment_id}.fact_daily_assigned` AS m
  WHERE m.experiment_id = __exp_id
    AND m.phase = 'test'
    AND m.metric_date BETWEEN __start AND __end
  GROUP BY m.user_id, m.variant, m.{dimension}
)
SELECT
  segment_value,
  variant,
  COUNT(*) AS assigned_users,
  SAFE_DIVIDE(SUM(metric_value), COUNT(*)) AS metric_per_user
FROM scoped
GROUP BY segment_value, variant
ORDER BY segment_value, variant
"""

BQ_SQL_ASSIGNMENT = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';

SELECT
  variant,
  COUNT(*) AS users,
  SAFE_DIVIDE(COUNT(*), SUM(COUNT(*)) OVER ()) AS share
FROM `analytics.ab.{experiment_id}.fact_user_assignments`
WHERE experiment_id = __exp_id
GROUP BY variant
ORDER BY variant
"""


def lift_analysis(
    rows: Sequence[Sequence[Any]],
    metric: MetricDef,
    *,
    control: str = "control",
    covariate_index: Optional[int] = None,
    expected_shares: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Per-user metric values -> lift, CI, p-value, optional CUPED.

    `rows` are already reduced to one row per assigned user:
    (user_id, experiment_id, variant, metric_value[, covariate])
    """
    from .stats import check_srm

    by_variant: Dict[str, List[Sequence[Any]]] = {}
    for r in rows:
        by_variant.setdefault(str(r[2]), []).append(r)

    if not by_variant:
        return {"error": "no rows"}

    if control not in by_variant:
        return {"error": f"control variant {control!r} not present"}

    observed = {v: len(rs) for v, rs in by_variant.items()}
    srm = None
    if expected_shares:
        srm = check_srm(observed, expected_shares).to_dict()

    # Control values are built in the lift loop below, after the population filter is
    # applied. Building them here as well would crash on NULL metric values and would
    # use an unfiltered population.

    out: Dict[str, Any] = {
        "metric": metric.name,
        "control": control,
        "users": sum(observed.values()),
        "variant_values": {},
        "lifts": {},
        "srm": srm,
    }

    for name, rs in sorted(by_variant.items()):
        # A NULL metric_value means "this user is not in the metric's population" —
        # for AOV, a non-purchaser. Those users are dropped from the average, not
        # counted as zero, because averaging AOV with zeros inserted is exactly how
        # AOV silently becomes revenue per user. This is a different rule from the
        # CUPED covariate, where NULL does mean zero.
        y = np.asarray(
            [float(r[3]) for r in rs if r[3] is not None], dtype=float
        )
        n_assigned = len(rs)
        out["variant_values"][name] = {
            "users": int(y.size),
            "assigned_users": int(n_assigned),
            "share": n_assigned / max(sum(observed.values()), 1),
            "metric_per_user": float(y.mean()) if y.size else float("nan"),
        }

    # Users outside the metric's population are excluded from every arm, not just the
    # one being reported, so both sides of the comparison share a population.
    in_pop = {name: [r for r in rs if r[3] is not None] for name, rs in by_variant.items()}
    ctrl = in_pop[control]
    y_c = np.asarray([float(r[3]) for r in ctrl], dtype=float)
    n_c = y_c.size

    for name, rs in sorted(by_variant.items()):
        if name == control:
            continue
        rs = in_pop[name]
        y_t = np.asarray([float(r[3]) for r in rs], dtype=float)
        n_t = y_t.size

        if metric.is_binary:
            res = _binary_lift(metric.name, y_c.sum(), y_t.sum(), n_c, n_t)
        else:
            res = _continuous_lift(metric.name, y_c, y_t)

        if covariate_index is not None and len(ctrl[0]) > covariate_index:
            t_c = np.asarray([0.0 if r[covariate_index] is None else float(r[covariate_index]) for r in ctrl])
            t_t = np.asarray([0.0 if r[covariate_index] is None else float(r[covariate_index]) for r in rs])
            cl, clo, chi_, cp = _cuped(metric.name, y_c, y_t, t_c, t_t)
            if cl is not None:
                res.cuped_relative_lift = float(cl)
                res.cuped_ci_low = None if clo is None else float(clo)
                res.cuped_ci_high = None if chi_ is None else float(chi_)
                res.cuped_p_value = None if cp is None else float(cp)

        out["lifts"][name] = res.to_dict()

    return out


def best_variant(lifts: Dict[str, Any], focus: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The variant with the highest lift whose CI excludes zero.

    "Highest lift" alone is the wrong question: the top point estimate is often the
    noisiest arm. Requiring a CI that excludes zero is what makes "best" defensible.
    Falls back to the highest point estimate when nothing is significant, and says so.

    `focus` restricts the comparison to one named arm, for "how did treatment_c do?".
    """
    usable = [
        (name, r) for name, r in lifts.items()
        if r.get("relative_lift") is not None and not math_isnan(r["relative_lift"])
    ]
    if focus:
        usable = [(n, r) for n, r in usable if n == focus] or usable
    if not usable:
        return None
    sig = [(n, r) for n, r in usable if r.get("significant") and r["relative_lift"] > 0]
    pool = sig or usable
    name, r = max(pool, key=lambda kv: kv[1]["relative_lift"])
    return {
        "variant": name,
        "lift": r["relative_lift"],
        "ci": [r["ci_low"], r["ci_high"]],
        "p_value": r["p_value"],
        "significant": bool(sig),
        "focused": bool(focus),
        "runner_up": _runner_up(name, usable),
    }


def _runner_up(name: str, usable: List[Any]) -> Optional[Dict[str, Any]]:
    others = [(n, r) for n, r in usable if n != name]
    if not others:
        return None
    n, r = max(others, key=lambda kv: kv[1]["relative_lift"])
    return {"variant": n, "lift": r["relative_lift"], "p_value": r["p_value"]}


def math_isnan(x: Any) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return True


def required_sample_size(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm sample size for a two-proportion test, normal approximation.

    Used to answer "is 90 days even long enough?" before anyone reads a lift number.
    """
    if not (0 < baseline_rate < 1) or mde <= 0:
        return -1
    p1 = baseline_rate * (1 + mde)
    p2 = baseline_rate
    pbar = (p1 + p2) / 2
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))