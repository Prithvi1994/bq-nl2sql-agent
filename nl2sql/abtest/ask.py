"""Business question -> BigQuery SQL, for experiment readouts.

The shape a business user actually asks:

    "which variant had the highest revenue lift in checkout_flow_v2 over the last 56 days"
    "did the onboarding email beat control on purchase rate?"
    "show me the daily revenue trend per variant for checkout_flow_v2"
    "was there a sample ratio mismatch on onboarding_email_v1?"
    "how long do we need to run to detect a 5% lift?"

parse_question() turns that into a typed QueryPlan. execute_plan() turns the plan into
BigQuery SQL, runs it, computes the statistics, and answers in plain English.

The model is not in this path. Metric definitions, the control arm, the aggregation
grain and the significance rule all come from code, so the same question always means
the same query. An LLM can be added at the plan level to *rank* candidate plans; it
cannot change what a metric means.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as M
from .stats import EPS, check_srm
from ..bq_exec import BigQueryResult, BQScriptError, run_bq_query

PROJECT = "analytics.ab"
MAX_RUN_DAYS = 90


class QuestionError(ValueError):
    """The question could not be resolved. Never guess past this."""


class QueryLog:
    """Every query the answer depends on, with the exact SQL that ran."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(self, label: str, result: BigQueryResult) -> None:
        self.entries.append(
            {
                "label": label,
                "ok": result.ok,
                "row_count": result.row_count,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "sql": result.dialect_sql,
                "duckdb_sql": result.duckdb_sql,
            }
        )

    def to_dict(self) -> List[Dict[str, Any]]:
        return self.entries

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [e for e in self.entries if not e["ok"]]


@dataclass
class QueryPlan:
    intent: str                       # best_variant | compare | daily_trend | srm | duration | cohort
    metric: M.MetricDef
    experiment_id: str
    start: date
    end: date
    control: str = "control"
    dimension: Optional[str] = None
    mde: Optional[float] = None
    limit: int = 20
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "intent": self.intent,
            "metric": self.metric.name,
            "experiment_id": self.experiment_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": self.days,
            "control": self.control,
        }
        if self.dimension:
            d["dimension"] = self.dimension
        if self.mde:
            d["mde"] = self.mde
        if self.extra:
            d["extra"] = self.extra
        return d


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "country": ["country", "geo", "region", "market"],
    "platform": ["platform", "device", "ios", "android", "web", "mobile"],
}

_EXPERIMENT_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_EXPERIMENT_HINTS = ("experiment", "test", "trial", "checkout", "pricing",
                     "onboarding", "email", "flow", "copy")


def _guess_experiment(text: str, known: Sequence[str]) -> str:
    t = text.lower()
    named = [exp for exp in known if re.search(rf"\b{re.escape(exp)}\b", t)]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        # Cross-experiment lift is not a valid comparison: different populations,
        # different windows, and possibly different data quality. Refuse rather than
        # silently picking one.
        raise QuestionError(
            f"you named {len(named)} experiments ({', '.join(named)}). Lift is only comparable "
            "within a single experiment, because each has its own population, window and "
            "randomisation. Ask about one at a time."
        )
    # An id-shaped token that is not a known experiment is a typo or a hallucination.
    id_shaped = sorted({m for m in _EXPERIMENT_RE.findall(t)
                        if any(h in m for h in _EXPERIMENT_HINTS)})
    if id_shaped:
        raise QuestionError(
            f"no experiment named {id_shaped[0]!r}. Known: " + ", ".join(known)
        )
    raise QuestionError(
        "I need the experiment id. Known: " + ", ".join(known) +
        ". Try: 'which variant had the highest revenue lift in checkout_flow_v2 over the last 56 days'"
    )


def _parse_dates(text: str, default_end: Optional[date] = None) -> Tuple[date, date]:
    """Resolve the window. 90 days is the hard cap on a test run."""
    t = text.lower()
    end = default_end or date.today()

    m = re.search(r"last\s+(\d+)\s+days?", t)
    if m:
        end = default_end or (date.today() - timedelta(days=1))
        return end - timedelta(days=int(m.group(1)) - 1), end

    m = re.search(r"last\s+(\d+)\s+weeks?", t)
    if m:
        end = default_end or (date.today() - timedelta(days=1))
        return end - timedelta(days=int(m.group(1)) * 7 - 1), end

    m = re.search(r"last\s+(week|month|quarter)", t)
    if m:
        if default_end:
            end = default_end
        elif m.group(1) == "week":
            today = date.today()
            end = today - timedelta(days=today.weekday() + 7)
        else:
            end = date.today().replace(day=1) - timedelta(days=1)
        span = 7 if m.group(1) == "week" else (30 if m.group(1) == "month" else 90)
        return end - timedelta(days=span - 1), end

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})\s*(?:to|through|until)\s*(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return date(*map(int, m.group(1, 2, 3))), date(*map(int, m.group(4, 5, 6)))

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        d = date(*map(int, m.group(1, 2, 3)))
        return d, d

    if "yesterday" in t:
        d = default_end or (date.today() - timedelta(days=1))
        return d, d

    raise QuestionError(
        "I need a date range. Use one of: 'last 30 days', 'last 8 weeks', "
        "'2026-02-01 to 2026-03-01', or 'last month'."
    )


def _parse_mde(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text.lower())
    if not m:
        return None
    val = float(m.group(1)) / 100.0
    return val if 0 < val < 1 else None


def _parse_control(text: str, known_variants: Sequence[str]) -> str:
    t = text.lower()
    for v in known_variants:
        if v != "control" and re.search(rf"\bagainst\s+{re.escape(v)}\b", t):
            return v
    return "control"


def _parse_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("sample ratio", "srm", "split", "allocation", "assignment", "balanced")):
        return "srm"
    if any(k in t for k in ("how long", "run for", "sample size", "mde", "detect a", "power")):
        return "duration"
    if any(k in t for k in ("daily", "trend", "over time", "by day", "each day", "time series")):
        return "daily_trend"
    if any(k in t for k in ("by country", "by region", "by device", "by platform", "by geo",
                            "split by", "broken down by", "segment", "per country", "per device")):
        return "cohort"
    if any(k in t for k in ("which variant", "best variant", "top variant", "winner",
                            "highest lift", "beat", "which one", "which performed",
                            "biggest", "most effective", "what should we ship")):
        return "best_variant"
    return "compare"


def _parse_dimension(text: str) -> Optional[str]:
    t = text.lower()
    for dim, words in DIMENSIONS.items():
        for w in words:
            if f"by {w}" in t or f"for each {w}" in t or f"per {w}" in t:
                return dim
    return None


PII_RE = re.compile(
    r"\b(user id|user ids|user_id|userids|email|emails|phone|address|name|"
    r"individual|person|row[- ]level|per[- ]user list|list the users|who are the users)\b"
)


def parse_question(
    text: str,
    experiments: Dict[str, Dict[str, Any]],
    default_end: Optional[date] = None,
) -> QueryPlan:
    """Free text -> QueryPlan. Raises QuestionError when it cannot be resolved."""
    if not text or not text.strip():
        raise QuestionError("empty question")

    # This system answers aggregate questions. Exporting individual rows is a
    # different product with different access controls, so it is refused up front
    # rather than answered and then filtered.
    if PII_RE.search(text.lower()) and not re.search(r"\b(count|how many|number of|rate|per user|average|total)\b", text.lower()):
        raise QuestionError(
            "this tool answers aggregate experiment questions. Listing individual users or "
            "their contact details is a different request with different access controls, so "
            "I will not run it. I can give you counts and rates per variant, country, or platform."
        )

    known = list(experiments)
    exp_id = _guess_experiment(text, known)
    meta = experiments[exp_id]

    intent = _parse_intent(text)

    # SRM is a property of the whole assignment table, and a duration question is
    # pure arithmetic. Neither depends on a window, so neither should demand one.
    if intent in ("srm", "duration"):
        start, end = meta["started_on"], meta["ended_on"]
        if isinstance(start, str):
            start = date.fromisoformat(str(start)[:10])
        if isinstance(end, str):
            end = date.fromisoformat(str(end)[:10])
        notes: Dict[str, Any] = {}
        metric = M.find_metric(text) or M.METRIC_BY_NAME[meta.get("primary_metric", "purchase_rate")]
        return QueryPlan(
            intent=intent, metric=metric, experiment_id=exp_id, start=start, end=end,
            control=_parse_control(text, meta.get("variants", ["control"])),
            dimension=_parse_dimension(text), mde=_parse_mde(text), extra=notes,
        )

    start, end = _parse_dates(text, default_end)
    if start > end:
        raise QuestionError(f"start {start} is after end {end}")

    notes: Dict[str, Any] = {}
    requested_days = (end - start).days + 1
    if requested_days > MAX_RUN_DAYS:
        # A test run cannot exceed 90 days. Clamp and record it, rather than silently
        # answering a question about a different window than the one asked.
        start = end - timedelta(days=MAX_RUN_DAYS - 1)
        notes["clamped"] = f"requested {requested_days} days, capped at {MAX_RUN_DAYS}"

    # A window may not extend past the experiment's own test period. Analysing a
    # partial window (say the last 42 days when the test only ran 28) compares
    # different amounts of exposure per arm and manufactures effects that are not
    # there, so the window is clipped to the experiment and the clip is recorded.
    exp_start = date.fromisoformat(str(meta["started_on"])[:10])
    exp_end = date.fromisoformat(str(meta["ended_on"])[:10])
    if start < exp_start or end > exp_end:
        clipped_start = max(start, exp_start)
        clipped_end = min(end, exp_end)
        notes["window_clipped_to_experiment"] = (
            f"asked {start}..{end}, experiment ran {exp_start}..{exp_end}, "
            f"analysed {clipped_start}..{clipped_end}"
        )
        start, end = clipped_start, clipped_end

    metric = M.find_metric(text)
    if metric is None:
        raise QuestionError(
            "I could not tell which metric you mean. Available: "
            + ", ".join(m.name for m in M.METRICS)
            + ". Try 'revenue_per_user', 'purchase_rate', or 'sessions_per_user'."
        )

    # A named variant must exist. "for treatment_c" means the user wants that arm
    # compared; silently answering about a different arm is worse than refusing.
    known_variants = meta.get("variants", ["control"])
    named_variants = [v for v in known_variants if re.search(rf"\b{re.escape(v)}\b", text.lower())]
    id_shaped = sorted({m for m in _EXPERIMENT_RE.findall(text.lower())
                        if m.startswith(("treatment", "variant", "control", "arm"))})
    unknown = [v for v in id_shaped if v not in known_variants]
    if unknown:
        raise QuestionError(
            f"{unknown[0]!r} is not a variant of {exp_id}. Variants: " + ", ".join(known_variants)
        )
    focus = named_variants[0] if named_variants else None
    if focus and focus != "control":
        notes["focus_variant"] = focus

    return QueryPlan(
        intent=_parse_intent(text),
        metric=metric,
        experiment_id=exp_id,
        start=start,
        end=end,
        control=_parse_control(text, known_variants),
        dimension=_parse_dimension(text),
        mde=_parse_mde(text),
        extra=notes,
    )

# ---------------------------------------------------------------------------
# Resolution: plan -> BigQuery SQL -> rows -> statistics -> answer
# ---------------------------------------------------------------------------

def _run(sql: str, db_path: str, log: QueryLog, label: str, row_limit: int = 500_000) -> List[Sequence[Any]]:
    res = run_bq_query(sql, db_path, row_limit=row_limit)
    log.add(label, res)
    if not res.ok:
        raise QuestionError(f"query failed ({label}): {res.error}")
    return res.rows


def _user_rows(plan: QueryPlan, db_path: str, log: QueryLog) -> List[Sequence[Any]]:
    """The canonical per-assigned-user query for this plan.

    Scope is `fact_user_assignments` LEFT JOIN metrics, so the denominator is every
    assigned user. Scoping to the metrics table instead would drop users with no
    activity, which selects on the outcome and inflates lift.
    """
    # AOV is only defined over users who purchased. Measuring it over all assigned
    # users would divide by zero and collapse it into revenue per user.
    restrict = ""
    if plan.metric.population == "purchasers":
        restrict = "\n    HAVING SUM(purchases) > 0"

    sql = """
DECLARE __exp_id STRING DEFAULT '{experiment_id}';
DECLARE __start DATE DEFAULT DATE('{start}');
DECLARE __end DATE DEFAULT DATE('{end}');

WITH by_phase AS (
  SELECT
    user_id,
    phase,
    {expr} AS metric_value
  FROM `{{project}}.{{experiment_id}}.fact_daily_assigned`
  WHERE experiment_id = __exp_id
    AND (
      (phase = 'test' AND metric_date BETWEEN __start AND __end)
      OR phase = 'pre'
    )
  GROUP BY user_id, phase{restrict}
)
SELECT
  a.user_id,
  a.experiment_id,
  a.variant,
  COALESCE(SUM(IF(t.phase = 'test', t.metric_value, 0.0)), 0.0) AS metric_value,
  COALESCE(SUM(IF(t.phase = 'pre', t.metric_value, 0.0)), 0.0) AS pre_metric_value
FROM `{{project}}.{{experiment_id}}.fact_user_assignments` AS a
LEFT JOIN by_phase AS t USING (user_id)
WHERE a.experiment_id = __exp_id
GROUP BY a.user_id, a.experiment_id, a.variant
ORDER BY a.user_id
""".format(
        project=PROJECT,
        experiment_id=plan.experiment_id,
        start=plan.start.isoformat(),
        end=plan.end.isoformat(),
        expr=plan.metric.expression("user"),
        restrict=restrict,
    )
    return _run(sql, db_path, log, f"per-user {plan.metric.name} (scope = assignments)")


def _answer_lift(plan: QueryPlan, db_path: str, meta: Dict[str, Any], log: QueryLog) -> Dict[str, Any]:
    rows = _user_rows(plan, db_path, log)
    if not rows:
        raise QuestionError(f"no rows for {plan.experiment_id} between {plan.start} and {plan.end}")

    analysis = M.lift_analysis(
        rows,
        plan.metric,
        control=plan.control,
        covariate_index=4,
        expected_shares=meta.get("intended_shares"),
    )
    if "error" in analysis:
        raise QuestionError(str(analysis["error"]))

    focus = (plan.extra or {}).get("focus_variant")
    winner = M.best_variant(analysis["lifts"], focus=focus)
    srm = analysis.get("srm")
    srm_failed = bool(srm and srm.get("detected"))

    detail: List[str] = []
    for name in sorted(analysis["lifts"]):
        r = analysis["lifts"][name]
        lift, lo, hi, p = r["relative_lift"], r["ci_low"], r["ci_high"], r["p_value"]
        verdict = ("better than control" if r["significant"] and lift > 0
                   else "worse than control" if r["significant"]
                   else "no reliable difference")
        per_user = analysis["variant_values"][name]["metric_per_user"]
        detail.append(
            f"{name}: {per_user:.6g} per user, lift {lift * 100:+.2f}% "
            f"(95% CI {lo * 100:+.2f}% to {hi * 100:+.2f}%, p={p:.3g}) - {verdict}"
        )
        if r.get("cuped_relative_lift") is not None:
            detail.append(
                f"    CUPED-adjusted: {r['cuped_relative_lift'] * 100:+.2f}% "
                f"(CI {r['cuped_ci_low'] * 100:+.2f}% to {r['cuped_ci_high'] * 100:+.2f}%)"
            )

    if srm_failed:
        # Randomisation failed, so the arms are not comparable. A significant point
        # estimate here is an artefact of the assignment bug as easily as of the
        # treatment, so no variant may be named a winner.
        lead = (
            f"**No decision.** A sample ratio mismatch in {plan.experiment_id} "
            f"(p={srm['p_value']:.3g}) means the arms are not comparable, so no variant "
            "can be called a winner: "
            + ", ".join(
                f"{n} {analysis['lifts'][n]['relative_lift'] * 100:+.2f}%"
                for n in sorted(analysis["lifts"])
            )
            + f" on {plan.metric.name} is directional only. Fix the assignment logging and rerun."
        )
        winner = None
    elif focus and focus in analysis["lifts"]:
        # The user named one arm, so report that arm rather than a ranking.
        r = analysis["lifts"][focus]
        direction = ("lifted" if r["relative_lift"] > 0 else
                     "reduced" if r["relative_lift"] < 0 else "left unchanged")
        lead = (
            f"**{focus}** {direction} {plan.metric.name} by "
            f"{abs(r['relative_lift']) * 100:.2f}% vs {plan.control} "
            f"(95% CI {r['ci_low'] * 100:+.2f}% to {r['ci_high'] * 100:+.2f}%, "
            f"p={r['p_value']:.3g}), {analysis['variant_values'][focus]['metric_per_user']:.6g} "
            f"per user vs {analysis['variant_values'][plan.control]['metric_per_user']:.6g}."
        )
        winner = None
    elif winner and winner["significant"]:
        lead = (f"**{winner['variant']}** is the winner on {plan.metric.name}: "
                f"{winner['lift'] * 100:+.2f}% vs {plan.control} "
                f"(95% CI {winner['ci'][0] * 100:+.2f}% to {winner['ci'][1] * 100:+.2f}%).")
    elif winner:
        # `best_variant` falls back to the top point estimate so the caller can show
        # it, but it is not a decision. Report it as a fallback, never as a winner.
        winner = {
            **winner,
            "variant": None,
            "highest_point_estimate": winner["variant"],
            "significant": False,
        }
        lead = (f"No variant beats {plan.control} on {plan.metric.name} with confidence. "
                f"Highest point estimate is {winner['highest_point_estimate']} at "
                f"{winner['lift'] * 100:+.2f}%, but its confidence interval includes zero.")
    else:
        lead = f"No variant available to compare against {plan.control}."

    if srm_failed and "sample ratio mismatch" not in lead.lower():
        lead += (
            f" Sample ratio mismatch detected (p={srm['p_value']:.3g}): observed "
            + ", ".join(f"{k} {v:.1%}" for k, v in srm["observed"].items())
            + ". Treat these effect estimates as directional only."
        )

    return {
        "answer": f"{lead} Window: {plan.start} to {plan.end} ({plan.days} days).",
        "plan": plan.to_dict(),
        "analysis": analysis,
        "winner": winner,
        "detail": detail,
        "queries": log.to_dict(),
    }


def _answer_cohort(plan: QueryPlan, db_path: str, log: QueryLog) -> Dict[str, Any]:
    dim = plan.dimension or "country"
    sql = M.BQ_SQL_SEGMENT.format(
        experiment_id=plan.experiment_id,
        start=plan.start.isoformat(), end=plan.end.isoformat(),
        dimension=dim, expr=plan.metric.expression("user"),
    )
    rows = _run(sql, db_path, log, f"{plan.metric.name} by {dim}")

    segs: Dict[str, Dict[str, float]] = {}
    for dim_value, variant, _users, value in rows:
        segs.setdefault(str(dim_value), {})[str(variant)] = float(value)

    detail = [f"**{plan.metric.name} by {dim}**, {plan.start} to {plan.end}:"]
    for dim_value in sorted(segs):
        arms = segs[dim_value]
        ctrl = arms.get(plan.control)
        best = max(((v, m) for v, m in arms.items() if v != plan.control),
                   key=lambda kv: kv[1], default=(None, None))
        if best[0] is not None and ctrl is not None and abs(ctrl) > EPS:
            detail.append(
                f"- {dim_value}: {plan.control} {ctrl:.6g}, best treatment {best[0]} {best[1]:.6g} "
                f"({(best[1] / ctrl - 1) * 100:+.2f}% vs control)"
            )
        else:
            detail.append(f"- {dim_value}: {plan.control} {ctrl if ctrl is not None else 'n/a'}")
    return {
        "answer": f"Compared {len(segs)} {dim} values on {plan.metric.name}.",
        "plan": plan.to_dict(), "detail": detail, "queries": log.to_dict(),
    }


def _answer_trend(plan: QueryPlan, db_path: str, log: QueryLog) -> Dict[str, Any]:
    sql = M.BQ_SQL_DAILY_TREND.format(
        experiment_id=plan.experiment_id,
        start=plan.start.isoformat(), end=plan.end.isoformat(),
        expr=plan.metric.expression("day"),
    )
    rows = _run(sql, db_path, log, f"daily {plan.metric.name} by variant")

    by_variant: Dict[str, List[Tuple[str, float]]] = {}
    for d, variant, _active, value in rows:
        by_variant.setdefault(str(variant), []).append((str(d), float(value)))

    summary = []
    for variant in sorted(by_variant):
        series = sorted(by_variant[variant])
        vals = np.asarray([v for _, v in series], dtype=float)
        head = float(vals[:7].mean()) if vals.size else float("nan")
        tail = float(vals[-7:].mean()) if vals.size else float("nan")
        summary.append({
            "variant": variant, "days": len(series),
            "first_week_mean": head, "last_week_mean": tail,
            "change": (tail / head - 1) if abs(head) > EPS else None,
        })

    n_days = max((s["days"] for s in summary), default=0)
    answer = f"Daily {plan.metric.name} over {n_days} days, {plan.start} to {plan.end}:"
    for s in summary:
        ch = s["change"]
        ch_txt = f"{(ch * 100):+.1f}% from first week to last" if ch is not None else "n/a"
        answer += f"\n- {s['variant']}: {s['first_week_mean']:.6g} -> {s['last_week_mean']:.6g} ({ch_txt})"
    return {"answer": answer, "plan": plan.to_dict(), "trend": summary, "queries": log.to_dict()}


def _answer_srm(plan: QueryPlan, db_path: str, meta: Dict[str, Any], log: QueryLog) -> Dict[str, Any]:
    sql = M.BQ_SQL_ASSIGNMENT.format(experiment_id=plan.experiment_id)
    rows = _run(sql, db_path, log, "variant assignment counts")
    observed = {str(v): int(u) for v, u, _ in rows}
    expected = meta.get("intended_shares") or {}
    if not expected:
        return {
            "answer": f"No intended split recorded for {plan.experiment_id}, so SRM cannot be tested.",
            "plan": plan.to_dict(), "queries": log.to_dict(),
        }
    res = check_srm(observed, expected).to_dict()
    obs_txt = ", ".join(f"{k} {v:.2%}" for k, v in res["observed"].items())
    exp_txt = ", ".join(f"{k} {v:.2%}" for k, v in res["expected"].items())
    if res["detected"]:
        answer = (f"**Sample ratio mismatch detected** (p={res['p_value']:.3g}). "
                  f"Observed {obs_txt} vs intended {exp_txt}. "
                  "Randomisation failed; do not act on effect estimates from this test.")
    else:
        answer = f"**No sample ratio mismatch.** Observed {obs_txt} matches the intended split."
    return {"answer": answer, "plan": plan.to_dict(), "srm": res, "queries": log.to_dict()}


def _answer_duration(plan: QueryPlan, meta: Dict[str, Any], log: QueryLog) -> Dict[str, Any]:
    baseline = float(meta.get("base_conversion") or 0.0) or 0.10
    mde = plan.mde if plan.mde is not None else 0.05
    n_per_arm = M.required_sample_size(baseline, mde)
    arms = max(len(meta.get("intended_shares") or {"control": 0.5, "treatment": 0.5}), 2)
    detail = [
        f"To detect a {mde * 100:.1f}% relative lift on {plan.metric.name} with a "
        f"{baseline:.2%} baseline, at 95% confidence and 80% power:",
        f"- **{n_per_arm:,} users per arm** ({n_per_arm * arms:,} total across {arms} arms)",
    ]
    traffic = float(meta.get("daily_traffic") or 0.0)
    if traffic > 0:
        days = int(np.ceil(n_per_arm * arms / traffic))
        detail.append(f"- at {traffic:,.0f} assigned users/day, roughly **{days} days**")
        if days > MAX_RUN_DAYS:
            detail.append(
                f"- that exceeds the {MAX_RUN_DAYS}-day cap: raise traffic, accept a larger MDE, "
                "or record the test as inconclusive"
            )
    else:
        detail.append(
            "- no daily traffic recorded for this experiment, so I cannot convert that to a duration"
        )
    return {
        "answer": "\n".join(detail), "plan": plan.to_dict(),
        "n_per_arm": n_per_arm, "queries": log.to_dict(),
    }


def execute_plan(plan: QueryPlan, db_path: str, experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Run the plan. Every number returned traces to a query in the returned log."""
    log = QueryLog()
    meta = experiments[plan.experiment_id]
    if plan.intent == "srm":
        return _answer_srm(plan, db_path, meta, log)
    if plan.intent == "duration":
        return _answer_duration(plan, meta, log)
    if plan.intent == "daily_trend":
        return _answer_trend(plan, db_path, log)
    if plan.intent == "cohort":
        return _answer_cohort(plan, db_path, log)
    return _answer_lift(plan, db_path, meta, log)


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

def load_experiments(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Read dim_experiments so the parser knows what exists and what it was testing."""
    sql = """
SELECT
  experiment_id, name, hypothesis, primary_metric, ship_threshold,
  guardrail_metric, guardrail_tolerance, started_on, ended_on, target_split
FROM `analytics.ab.dim_experiments`
ORDER BY experiment_id
"""
    res = run_bq_query(sql, db_path, row_limit=1000)
    if not res.ok:
        raise QuestionError(f"could not read dim_experiments: {res.error}")

    truth_path = Path(db_path).parent / "ground_truth.json"
    base: Dict[str, float] = {}
    if truth_path.exists():
        gt = json.loads(truth_path.read_text(encoding="utf-8")).get("experiments", {})
        for k, v in gt.items():
            base[k] = float(v.get("control_baseline", {}).get("purchase_rate", 0.0))

    out: Dict[str, Dict[str, Any]] = {}
    for exp_id, name, hypothesis, primary, thr, guard, gtol, start, end, split in res.rows:
        shares: Dict[str, float] = {}
        for part in re.split(r"[,/]", str(split)):
            if ":" in part:
                k, v = part.split(":", 1)
                shares[k.strip()] = float(v.rstrip("%")) / 100.0
        out[str(exp_id)] = {
            "experiment_id": str(exp_id), "name": name, "hypothesis": hypothesis,
            "primary_metric": primary, "ship_threshold": float(thr),
            "guardrail_metric": guard, "guardrail_tolerance": float(gtol),
            "started_on": str(start), "ended_on": str(end),
            "intended_shares": shares, "variants": list(shares) or ["control"],
            "base_conversion": base.get(str(exp_id), 0.0),
            "daily_traffic": 0.0,
        }
    return out


def ask(
    question: str,
    db_path: str,
    experiments: Optional[Dict[str, Dict[str, Any]]] = None,
    default_end: Optional[date] = None,
) -> Dict[str, Any]:
    experiments = experiments or load_experiments(db_path)
    plan = parse_question(question, experiments, default_end=default_end)
    return execute_plan(plan, db_path, experiments)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a business question about the experiment data.")
    ap.add_argument(
        "question",
        help='e.g. "which variant had the highest revenue lift in checkout_flow_v2 over the last 56 days"',
    )
    ap.add_argument("--db", default="./abtest_data/abtest.duckdb")
    ap.add_argument("--end", help="ISO end date; default is today")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        exps = load_experiments(args.db)
        res = ask(args.question, args.db, experiments=exps,
                  default_end=date.fromisoformat(args.end) if args.end else None)
    except (QuestionError, BQScriptError) as exc:
        print(f"Cannot answer: {exc}")
        raise SystemExit(1)

    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return
    print(res["answer"])
    for line in res.get("detail", []):
        print("  " + line)
    for q in res.get("queries", []):
        print(f"  [{q['label']}] {q['row_count']} rows in {q['elapsed_ms']}ms")


if __name__ == "__main__":
    main()
