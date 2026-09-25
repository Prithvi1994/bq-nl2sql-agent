"""The tool surface the experiment agent can call.

Every tool is a named, parameterised operation that runs real BigQuery SQL and returns
numbers. The model selects tools and reads their output; it never writes SQL, never
computes a statistic, and never names a variant as a winner. Those decisions are made
in code from the same rows the tool returned.

The tool list is deliberately small and fully typed. A hallucinated argument is
rejected by the schema check, so the failure mode is a clear error rather than a
plausible wrong query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import metrics as M
from .ask import MAX_RUN_DAYS, PROJECT, QuestionError, QueryLog
from .stats import EPS, check_srm
from ..bq_exec import BigQueryResult, run_bq_query


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, str]
    required: List[str] = field(default_factory=list)

    def signature(self) -> str:
        args = ", ".join(
            f"{k}: {v}" + ("" if k in self.required else " = optional")
            for k, v in self.parameters.items()
        )
        return f"{self.name}({args})"

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Reject unknown args and missing required args. No silent coercion."""
        unknown = set(args) - set(self.parameters)
        if unknown:
            raise QuestionError(
                f"unknown argument(s) {sorted(unknown)} for {self.name}; "
                f"valid: {sorted(self.parameters)}"
            )
        missing = [k for k in self.required if k not in args or args[k] is None]
        if missing:
            raise QuestionError(f"{self.name} is missing required argument(s): {missing}")
        return args


TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="list_experiments",
        description="List every experiment with its hypothesis, dates, variants, primary metric and intended split.",
        parameters={},
    ),
    ToolSpec(
        name="metric_definitions",
        description="Return the exact BigQuery expression, unit and population for every available metric.",
        parameters={},
    ),
    ToolSpec(
        name="variant_lift",
        description=(
            "Compute lift versus control for every treatment variant, with 95% confidence "
            "intervals, p-values, SRM check and the ship-rule verdict. Use this for any "
            "'which variant won' or 'how much lift' question. The scope is assigned users, "
            "not active users."
        ),
        parameters={
            "experiment_id": "string",
            "metric": "string, one of the names from metric_definitions",
            "start": "ISO date, optional (defaults to the experiment start)",
            "end": "ISO date, optional (defaults to the experiment end)",
        },
        required=["experiment_id", "metric"],
    ),
    ToolSpec(
        name="variant_trend",
        description="Daily per-variant values of a metric across the test window, for charts and for checking whether an effect was stable or drifted.",
        parameters={
            "experiment_id": "string",
            "metric": "string",
            "start": "ISO date, optional",
            "end": "ISO date, optional",
        },
        required=["experiment_id", "metric"],
    ),
    ToolSpec(
        name="segment_lift",
        description="Lift versus control broken down by a dimension (country or platform), to find where an effect is concentrated.",
        parameters={
            "experiment_id": "string",
            "metric": "string",
            "dimension": "string, one of: country, platform",
            "start": "ISO date, optional",
            "end": "ISO date, optional",
        },
        required=["experiment_id", "metric", "dimension"],
    ),
    ToolSpec(
        name="check_srm",
        description="Chi-square test of the observed variant split against the intended split. Run this before trusting any effect estimate.",
        parameters={"experiment_id": "string"},
        required=["experiment_id"],
    ),
    ToolSpec(
        name="required_sample_size",
        description="Users per arm needed to detect a given relative lift at a given baseline rate, and whether the experiment ran long enough.",
        parameters={
            "experiment_id": "string",
            "mde": "float, relative MDE as a fraction, e.g. 0.05 for 5%",
        },
        required=["experiment_id", "mde"],
    ),
    ToolSpec(
        name="data_quality",
        description="Row counts, date coverage, inactive-user share and the experiment's pre-period size.",
        parameters={"experiment_id": "string"},
        required=["experiment_id"],
    ),
]

TOOL_BY_NAME = {t.name: t for t in TOOL_SPECS}


def catalog() -> str:
    return "\n".join(f"- {t.signature()}\n    {t.description}" for t in TOOL_SPECS)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class ToolBox:
    """Executes tools against the offline mirror. Every call records its SQL."""

    def __init__(self, db_path: str, experiments: Dict[str, Dict[str, Any]]) -> None:
        self.db_path = db_path
        self.experiments = experiments
        self.log = QueryLog()

    # -- helpers ----------------------------------------------------------
    def _exp(self, experiment_id: str) -> Dict[str, Any]:
        if experiment_id not in self.experiments:
            raise QuestionError(
                f"no experiment {experiment_id!r}. Available: {', '.join(self.experiments)}"
            )
        return self.experiments[experiment_id]

    def _window(
        self, meta: Dict[str, Any], start: Optional[str], end: Optional[str]
    ) -> tuple[date, date, List[str]]:
        s = date.fromisoformat(str(start)[:10]) if start else date.fromisoformat(str(meta["started_on"])[:10])
        e = date.fromisoformat(str(end)[:10]) if end else date.fromisoformat(str(meta["ended_on"])[:10])
        notes: List[str] = []
        if s > e:
            raise QuestionError(f"start {s} is after end {e}")
        if (e - s).days + 1 > MAX_RUN_DAYS:
            s = e - timedelta(days=MAX_RUN_DAYS - 1)
            notes.append(f"window capped at {MAX_RUN_DAYS} days")
        es = date.fromisoformat(str(meta["started_on"])[:10])
        ee = date.fromisoformat(str(meta["ended_on"])[:10])
        if s < es or e > ee:
            s, e = max(s, es), min(e, ee)
            notes.append(f"window clipped to the experiment's test period {s}..{e}")
        return s, e, notes

    def _metric(self, name: str) -> M.MetricDef:
        m = M.METRIC_BY_NAME.get(name)
        if m is None:
            raise QuestionError(
                f"unknown metric {name!r}. Available: {', '.join(M.METRIC_BY_NAME)}"
            )
        return m

    def _run(self, sql: str, label: str) -> BigQueryResult:
        res = run_bq_query(sql, self.db_path, row_limit=500_000)
        self.log.add(label, res)
        if not res.ok:
            raise QuestionError(f"{label} failed: {res.error}")
        return res

    # -- tools ------------------------------------------------------------
    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = TOOL_BY_NAME.get(name)
        if spec is None:
            raise QuestionError(
                f"unknown tool {name!r}. Available: {', '.join(TOOL_BY_NAME)}"
            )
        spec.validate(args or {})
        return getattr(self, f"_tool_{name}")(args or {})

    def _tool_list_experiments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "experiments": [
                {
                    "experiment_id": m["experiment_id"],
                    "name": m["name"],
                    "hypothesis": m["hypothesis"],
                    "started_on": m["started_on"],
                    "ended_on": m["ended_on"],
                    "variants": m["variants"],
                    "primary_metric": m["primary_metric"],
                    "ship_threshold": m["ship_threshold"],
                    "guardrail_metric": m["guardrail_metric"],
                    "guardrail_tolerance": m["guardrail_tolerance"],
                    "intended_shares": m["intended_shares"],
                }
                for m in self.experiments.values()
            ]
        }

    def _tool_metric_definitions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "metrics": [
                {
                    "name": m.name,
                    "category": m.category,
                    "description": m.description,
                    "population": m.population,
                    "bigquery_per_user": m.expression("user"),
                    "bigquery_per_day": m.expression("day"),
                }
                for m in M.METRICS
            ]
        }

    def _user_metric_sql(self, plan_exp: str, metric: M.MetricDef, s: date, e: date) -> str:
        restrict = "\n    HAVING SUM(purchases) > 0" if metric.population == "purchasers" else ""
        return """
DECLARE __exp_id STRING DEFAULT '{exp}';
DECLARE __start DATE DEFAULT DATE('{start}');
DECLARE __end DATE DEFAULT DATE('{end}');

WITH by_phase AS (
  SELECT
    user_id,
    phase,
    {expr} AS metric_value
  FROM `{project}.{exp}.fact_daily_assigned`
  WHERE experiment_id = __exp_id
    AND ((phase = 'test' AND metric_date BETWEEN __start AND __end) OR phase = 'pre')
  GROUP BY user_id, phase{restrict}
)
SELECT
  a.user_id,
  a.variant,
  COALESCE(SUM(IF(t.phase = 'test', t.metric_value, 0.0)), 0.0) AS metric_value,
  COALESCE(SUM(IF(t.phase = 'pre', t.metric_value, 0.0)), 0.0) AS pre_metric_value
FROM `{project}.{exp}.fact_user_assignments` AS a
LEFT JOIN by_phase AS t USING (user_id)
WHERE a.experiment_id = __exp_id
GROUP BY a.user_id, a.variant
ORDER BY a.user_id
""".format(project=PROJECT, exp=plan_exp, start=s.isoformat(), end=e.isoformat(),
           expr=metric.expression("user"), restrict=restrict)

    def _tool_variant_lift(self, args: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._exp(args["experiment_id"])
        metric = self._metric(args["metric"])
        s, e, notes = self._window(meta, args.get("start"), args.get("end"))
        res = self._run(
            self._user_metric_sql(meta["experiment_id"], metric, s, e),
            f"variant_lift: {metric.name} for {meta['experiment_id']}",
        )
        analysis = M.lift_analysis(
            res.rows, metric, control="control", covariate_index=3,
            expected_shares=meta.get("intended_shares"),
        )
        if "error" in analysis:
            raise QuestionError(str(analysis["error"]))
        analysis["window"] = {"start": s.isoformat(), "end": e.isoformat(), "days": (e - s).days + 1}
        analysis["notes"] = notes
        analysis["verdict"] = self._verdict(meta, metric, analysis)
        return analysis

    def _verdict(self, meta: Dict[str, Any], metric: M.MetricDef, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """The pre-registered decision rule, applied in code.

        Ship requires: no SRM, the primary metric clears the threshold with a CI that
        excludes zero, and the guardrail holds. Anything else is a hold, not a ship.
        """
        srm = analysis.get("srm") or {}
        if srm.get("detected"):
            return {
                "decision": "HOLD",
                "reason": f"sample ratio mismatch (p={srm['p_value']:.3g}); arms are not comparable",
                "shipping_variant": None,
            }
        winner = M.best_variant(analysis["lifts"])
        if winner is None or not winner.get("significant"):
            return {
                "decision": "NO_SHIP",
                "reason": "no variant shows a reliable difference from control on this metric",
                "shipping_variant": None,
            }
        if metric.name != meta.get("primary_metric"):
            return {
                "decision": "HOLD",
                "reason": f"{metric.name} is not the primary metric ({meta.get('primary_metric')})",
                "shipping_variant": None,
            }
        if winner["lift"] < meta.get("ship_threshold", 0.0):
            return {
                "decision": "NO_SHIP",
                "reason": (
                    f"lift {winner['lift']:+.2%} does not clear the "
                    f"{meta.get('ship_threshold', 0.0):.2%} ship threshold"
                ),
                "shipping_variant": None,
            }
        guard_name = meta.get("guardrail_metric")
        guard_tol = float(meta.get("guardrail_tolerance", 0.0))
        if guard_name in analysis["lifts"]:
            g = analysis["lifts"][guard_name][winner["variant"]]
            if g["relative_lift"] < guard_tol:
                return {
                    "decision": "NO_SHIP",
                    "reason": (
                        f"primary clears but {guard_name} regressed {g['relative_lift']:+.2%}, "
                        f"outside the {guard_tol:+.2%} guardrail"
                    ),
                    "shipping_variant": None,
                }
        return {
            "decision": "SHIP",
            "reason": (
                f"{winner['variant']} clears the threshold on {metric.name} "
                f"({winner['lift']:+.2%}) and the {guard_name} guardrail holds"
            ),
            "shipping_variant": winner["variant"],
        }

    def _tool_variant_trend(self, args: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._exp(args["experiment_id"])
        metric = self._metric(args["metric"])
        s, e, notes = self._window(meta, args.get("start"), args.get("end"))
        sql = M.BQ_SQL_DAILY_TREND.format(
            experiment_id=meta["experiment_id"], start=s.isoformat(), end=e.isoformat(),
            expr=metric.expression("day"),
        )
        res = self._run(sql, f"daily {metric.name} for {meta['experiment_id']}")
        by_variant: Dict[str, List[Dict[str, Any]]] = {}
        for d, variant, active, value in res.rows:
            by_variant.setdefault(str(variant), []).append(
                {"date": str(d), "active_users": int(active), "value": float(value)}
            )
        for series in by_variant.values():
            series.sort(key=lambda r: r["date"])
        return {"metric": metric.name, "window": {"start": s.isoformat(), "end": e.isoformat()},
                "notes": notes, "series": by_variant}

    def _tool_segment_lift(self, args: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._exp(args["experiment_id"])
        metric = self._metric(args["metric"])
        dim = str(args["dimension"])
        if dim not in ("country", "platform"):
            raise QuestionError(f"unknown dimension {dim!r}; use country or platform")
        s, e, notes = self._window(meta, args.get("start"), args.get("end"))
        sql = M.BQ_SQL_SEGMENT.format(
            experiment_id=meta["experiment_id"], start=s.isoformat(), end=e.isoformat(),
            dimension=dim, expr=metric.expression("user"),
        )
        res = self._run(sql, f"{metric.name} by {dim} for {meta['experiment_id']}")
        segs: Dict[str, Dict[str, float]] = {}
        for dim_value, variant, users, value in res.rows:
            segs.setdefault(str(dim_value), {})[str(variant)] = float(value)
        rows = []
        for dim_value in sorted(segs):
            arms = segs[dim_value]
            ctrl = arms.get("control")
            treatments = {v: m for v, m in arms.items() if v != "control"}
            best = max(treatments.items(), key=lambda kv: kv[1], default=(None, None))
            rows.append({
                "segment": dim_value,
                "control": ctrl,
                "best_treatment": best[0],
                "best_treatment_value": best[1],
                "relative_to_control": (best[1] / ctrl - 1) if best[1] is not None and ctrl else None,
                "all_variants": arms,
            })
        return {"metric": metric.name, "dimension": dim, "notes": notes, "segments": rows}

    def _tool_check_srm(self, args: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._exp(args["experiment_id"])
        sql = M.BQ_SQL_ASSIGNMENT.format(experiment_id=meta["experiment_id"])
        res = self._run(sql, f"assignment counts for {meta['experiment_id']}")
        observed = {str(v): int(u) for v, u, _ in res.rows}
        expected = meta.get("intended_shares") or {}
        if not expected:
            return {"experiment_id": meta["experiment_id"], "error": "no intended split recorded"}
        out = check_srm(observed, expected).to_dict()
        out["experiment_id"] = meta["experiment_id"]
        out["users"] = sum(observed.values())
        return out

    def _tool_required_sample_size(self, args: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._exp(args["experiment_id"])
        mde = float(args["mde"])
        if not 0 < mde < 1:
            raise QuestionError(f"mde must be a fraction between 0 and 1, got {mde}")
        baseline = float(meta.get("base_conversion") or 0.0) or 0.10
        n = M.required_sample_size(baseline, mde)
        arms = max(len(meta.get("intended_shares") or {"control": 0.5, "treatment": 0.5}), 2)
        sql = M.BQ_SQL_ASSIGNMENT.format(experiment_id=meta["experiment_id"])
        res = self._run(sql, f"assignment counts for {meta['experiment_id']}")
        actual = {str(v): int(u) for v, u, _ in res.rows}
        return {
            "experiment_id": meta["experiment_id"],
            "mde": mde,
            "baseline_rate": baseline,
            "users_per_arm_required": n,
            "total_required": n * arms,
            "users_per_arm_actual": actual,
            "powered": bool(actual) and all(c >= n for c in actual.values()),
            "verdict": (
                "the experiment was powered to detect this MDE"
                if actual and all(c >= n for c in actual.values())
                else "the experiment was underpowered for this MDE"
            ),
        }

    def _tool_data_quality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._exp(args["experiment_id"])
        exp = meta["experiment_id"]
        sql = f"""
SELECT
  phase,
  COUNT(*) AS rows,
  COUNT(DISTINCT user_id) AS users,
  MIN(metric_date) AS first_date,
  MAX(metric_date) AS last_date,
  SUM(sessions) AS sessions,
  SUM(purchases) AS purchases,
  SUM(revenue_usd) AS revenue_usd
FROM `{PROJECT}.{exp}.fact_daily_user_metrics`
WHERE experiment_id = '{exp}'
GROUP BY phase
ORDER BY phase
"""
        res = self._run(sql, f"data quality for {exp}")
        phases = {
            str(phase): {
                "rows": int(rows), "users": int(users),
                "first_date": str(first), "last_date": str(last),
                "sessions": int(sessions), "purchases": int(purchases),
                "revenue_usd": round(float(revenue), 2),
            }
            for phase, rows, users, first, last, sessions, purchases, revenue in res.rows
        }
        assigned = sum(1 for _ in ()) or None
        asg = self._run(
            f"SELECT COUNT(*) FROM `{PROJECT}.{exp}.fact_user_assignments` WHERE experiment_id = '{exp}'",
            f"assignment count for {exp}",
        )
        n_assigned = int(asg.rows[0][0]) if asg.rows else 0
        test_users = phases.get("test", {}).get("users", 0)
        return {
            "experiment_id": exp,
            "assigned_users": n_assigned,
            "users_with_test_activity": test_users,
            "inactive_share": round(1 - test_users / n_assigned, 4) if n_assigned else None,
            "phases": phases,
            "caveats": [
                f"{n_assigned - test_users} assigned users had no test-period activity; "
                "they are included in every lift denominator"
            ] if n_assigned and test_users < n_assigned else [],
        }
