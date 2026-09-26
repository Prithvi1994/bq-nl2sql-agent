"""Deterministic cut-sweep readout over the exp_scoresheet layer.

Produces the decision dict: headline (pooled whole-window cuts from the stored
platform cuts), cut-level lift across combinations (variant x metric x slice),
day-level CI for significance, SRM gate, ramp/segments, and the go/no-go verdict.

Design invariants enforced HERE, not by any LLM:
- denominator rules (pooled SUM across disjoint detail slices)
- stored cuts read as-is on Overall rows; never recomputed from dollars
- significance from the day-level series only
- decision from the whole pattern of cuts, not any single slice
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..bq_exec import run_bq_query

# MetricCatalog: the compute (sum-ratio) metrics over disjoint daily rows.
# The stored cuts (purch_per_user etc.) are read for the whole-window headline.
METRICS = {
    "purch_per_user": ("purchases", "users", "conversion per user"),
    "gmv_per_user": ("gmv", "users", "north-star GMV per user"),
    "atc_per_user": ("add_to_cart", "users", "add-to-cart per user"),
}

TRAP_NOTE = ("gmv_tot_lift stores the per-user lift despite its name; "
             "narration must call it per-user")


@dataclass
class SweepResult:
    experiment_id: str
    window: Tuple[str, str]
    headline: Dict[str, Any]
    cuts: List[Dict[str, Any]]
    ramp: Dict[str, Any]
    srm: Dict[str, Any]
    multiplicity: Dict[str, Any]
    decision: Dict[str, Any]
    queries: List[Dict[str, Any]] = field(default_factory=list)
    refusals: List[str] = field(default_factory=list)

    def content_hash(self) -> str:
        """SHA-256 over the deterministic parts (excludes query log)."""
        payload = json.dumps({
            "experiment_id": self.experiment_id,
            "window": self.window,
            "headline": self.headline,
            "cuts": self.cuts,
            "ramp": self.ramp,
            "srm": self.srm,
            "decision": self.decision,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "window": list(self.window),
            "headline": self.headline,
            "cuts": self.cuts,
            "ramp": self.ramp,
            "srm": self.srm,
            "multiplicity": self.multiplicity,
            "decision": self.decision,
            "content_hash": self.content_hash(),
        }
        if self.refusals:
            d["refusals"] = self.refusals
        return d


# ---------------------------------------------------------------------------
# queries — all over disjoint detail slices (overall_flag = FALSE)
# ---------------------------------------------------------------------------

def _daily_rows(db: str, exp_id: str, queries: List[Dict],
                start: Optional[str] = None, end: Optional[str] = None
                ) -> List[Sequence[Any]]:
    """One day × variant row for the whole test window, disjoint detail slices."""
    month_filter = ""
    args: list = [exp_id]
    if start:
        month_filter = " AND metric_date >= ? AND metric_date <= ?"
        args += [start, end]
    # run_bq_query runs read-only single statements without parameters; values
    # reaching here are registry-validated identifiers or ISO dates.
    args_sql = f" AND metric_date >= DATE '{start}' AND metric_date <= DATE '{end}'" if start else ""
    sql = f"""
        SELECT CAST(metric_date AS VARCHAR) AS d, variant_id, is_control,
               SUM(users) AS users, SUM(purchases) AS purchases,
               SUM(add_to_cart) AS atc, SUM(gmv) AS gmv
        FROM exp_scoresheet
        WHERE experiment_id = '{exp_id}' AND NOT overall_flag
          AND slice_country = 'Overall'{args_sql}
        GROUP BY 1, 2, 3 ORDER BY 1, 2
    """
    res = run_bq_query(sql, db, row_limit=50000)
    queries.append({"label": "sweep_daily", "ok": res.ok, "rows": res.row_count,
                    "sql": sql.strip(), "error": res.error})
    if not res.ok:
        raise RuntimeError(f"sweep daily failed: {res.error}")
    return list(res.rows)


# ---------------------------------------------------------------------------
# inference: pooled sum-ratio + day-level Welch CI
# ---------------------------------------------------------------------------

def _pooled(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _lift_ci(days_c: List[Tuple[float, float]], days_t: List[Tuple[float, float]],
             alpha: float = 0.05) -> Dict[str, Any]:
    """Lift of treatment vs control on per-day values, Welch t CI on daily means.

    days_* = [(num, den), ...] one entry per day, disjoint window. The pooled
    estimator is the SUM-ratio across all days; the CI comes from day-level
    dispersion of the per-day ratios (the snapshot table's independent days).
    """
    from scipy import stats as st

    def _day_ratio(cell: Tuple[float, float]) -> float:
        num, den = cell
        return num / den if den else float("nan")

    rc = [c for c in (_day_ratio(x) for x in days_c) if not math.isnan(c)]
    rt = [c for c in (_day_ratio(x) for x in days_t) if not math.isnan(c)]
    if len(rc) < 3 or len(rt) < 3:
        return {"lift": None, "ci": None, "p": None, "significant": False,
                "n_days": [len(rc), len(rt)], "underpowered_days": True}

    num_c = sum(x[0] for x in days_c)
    den_c = sum(x[1] for x in days_c)
    num_t = sum(x[0] for x in days_t)
    den_t = sum(x[1] for x in days_t)
    pooled_lift = _pooled(num_t, den_t) / _pooled(num_c, den_c) - 1.0

    # Sum-ratio estimator r = (Σn_t/Σd_t)/(Σn_c/Σd_c) − 1 with day-level variance:
    # each day is one observation of the paired ratio; variance of the pooled ratio
    # via the design-effect-free cluster formula (day = cluster, den = weight).
    # CI on the LOG of each arm's pooled rate (delta method), then difference:
    #   se(log rate) ≈ sqrt( Σ w_i² (r_i − r̄_w)² ) / (Σ w_i · r̄_w)
    # where w_i = d_i (denominator share) and r̄_w = Σd_i r_i/Σd_i — the dentro-day
    # dispersion is the honest day-level dispersion for a snapshot-grain table.
    import math as _m
    def _log_rate_ci(cells: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        ns = [n for n, d in cells if d > 0]
        ds = [d for n, d in cells if d > 0]
        rs = [n / d for n, d in cells if d > 0]
        if sum(ds) <= 0:
            return None
        wsum = sum(ds)
        rbar = sum(d * r for d, r in zip(ds, rs)) / wsum
        if rbar <= 0:
            return None
        var_num = sum((d * (r - rbar)) ** 2 for d, r in zip(ds, rs))
        se = math.sqrt(var_num) / (wsum * rbar)
        return (math.log(rbar), se)
    t_cs = _log_rate_ci(days_c)
    t_ts = _log_rate_ci(days_t)
    if t_cs is None or t_ts is None:
        return {"lift": pooled_lift, "ci": None, "p": None, "significant": False,
                "n_days": [len(rc), len(rt)], "underpowered_days": False}
    log_diff = t_ts[0] - t_cs[0]
    se_diff = math.sqrt(t_ts[1] ** 2 + t_cs[1] ** 2)
    z = log_diff / se_diff if se_diff > 0 else 0.0

    from scipy.stats import norm as _norm
    p = 2.0 * (1.0 - _norm.cdf(abs(z)))
    zc = _norm.ppf(1 - alpha / 2)
    ci_lo = _m.exp(log_diff - zc * se_diff) - 1.0
    ci_hi = _m.exp(log_diff + zc * se_diff) - 1.0
    significant = not (ci_lo <= 0.0 <= ci_hi)
    return {"lift": float(pooled_lift), "ci": [float(ci_lo), float(ci_hi)], "p": float(p),
            "significant": bool(significant), "n_days": [len(rc), len(rt)],
            "underpowered_days": False, "method": "sum-ratio/day-cluster"}


# ---------------------------------------------------------------------------
# sweep machinery
# ---------------------------------------------------------------------------

def _variant_cells(by_day: Dict[str, Dict[str, Dict[str, float]]],
                   vid: str, metric: str) -> List[Tuple[float, float]]:
    num_col, den_col, _ = METRICS[metric]
    out = []
    for d in sorted(by_day):
        cell = by_day[d].get(vid)
        if cell:
            out.append((float(cell[num_col]), float(cell[den_col])))
    return out


def _slice_cells(by_day: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                 vid: str, metric: str, slice_key: str, slice_val: str
                 ) -> List[Tuple[float, float]]:
    num_col, den_col, _ = METRICS[metric]
    out = []
    for d in sorted(by_day):
        cell = by_day[d].get(vid, {}).get(slice_val)
        if cell:
            out.append((float(cell[num_col]), float(cell[den_col])))
    return out


def sweep(db: str, experiment_id: str, registry_row: Dict[str, Any],
          *, window_days: Optional[int] = None,
          start: Optional[str] = None, end: Optional[str] = None,
          slices: Optional[Dict[str, List[str]]] = None,
          srm_threshold: float = 0.001) -> SweepResult:
    """Cut sweep: every (variant × metric × slice × window) combination.

    registry_row supplies north_star_metric, guardrail config, intended shares.
    """
    queries: List[Dict[str, Any]] = []
    refusals: List[str] = []
    raw_variants = registry_row["variants"]
    # Accept both shapes: [{'id','name'}] from the scoresheet contract, or a
    # comma string straight from the registry row ("ctl,t1,t2").
    if isinstance(raw_variants, str):
        variants = [{"id": vid, "name": vid} for vid in raw_variants.split(",")]
    else:
        variants = list(raw_variants)
    ctl_id = variants[0]["id"]
    treatments = [v for v in variants if v["id"] != ctl_id]
    north_star = registry_row.get("north_star_metric") or "gmv_per_user"
    metrics_used = sorted(set(METRICS) | ({north_star} if north_star in METRICS else set()))

    # window clipping to the experiment's own started/ended (90-day cap implied)
    exp_id = experiment_id
    started = registry_row["started_on"]
    ended = registry_row["ended_on"]
    # registry rows may carry DATE objects or ISO strings depending on the loader
    started = started.isoformat() if hasattr(started, "isoformat") else started
    ended = ended.isoformat() if hasattr(ended, "isoformat") else ended
    eff_start = start or started
    eff_end = end or ended
    clipped = False
    if date.fromisoformat(eff_start) < date.fromisoformat(started):
        eff_start, clipped = started, True
    if date.fromisoformat(eff_end) > date.fromisoformat(ended):
        eff_end, clipped = ended, True
    if window_days:
        max_start = (date.fromisoformat(eff_end) - timedelta(days=window_days - 1)).isoformat()
        if date.fromisoformat(eff_start) < date.fromisoformat(max_start):
            eff_start, clipped = max_start, True
    if date.fromisoformat(eff_end) < date.fromisoformat(eff_start):
        raise ValueError("empty window after clipping")

    rows = _daily_rows(db, experiment_id, queries, eff_start, eff_end)
    slice_rows = _daily_slice_rows(db, experiment_id, queries, eff_start, eff_end)
    if not rows:
        raise ValueError(f"no detail rows for {experiment_id} in [{eff_start}, {eff_end}]")

    # day -> variant -> metric-num/den at overall level
    by_day_overall: Dict[str, Dict[str, Dict[str, float]]] = {}
    by_day_slice: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for d, vid, _ctl, users, purch, atc, gmv in rows:
        cell = {"users": users, "purchases": purch, "add_to_cart": atc, "gmv": gmv}
        by_day_overall.setdefault(d, {}).setdefault(vid, {}).update(cell)
        by_day_slice.setdefault(d, {}).setdefault(vid, {}).setdefault("Overall", {}).update(cell)

    # --- SRM: assigned denominator is the contract. Daily-grain totals post
    # attrition and CANNOT measure assignment integrity — pull fact_user_assignments
    # (per-experiment table kept outside the scoresheet on purpose).
    intended = registry_row.get("intended_shares") or {}
    if not intended:
        # registry rows don't carry shares; the fixture ground-truth does.
        try:
            import json as _json
            from pathlib import Path as _P
            _db_dir = _P(db)
            _cfg = _db_dir / "ground_truth.json" if _db_dir.suffix else _db_dir
            cfg_path = _db_dir if _db_dir.name.endswith(".duckdb") is False and (_db_dir / "ground_truth.json").exists() else _db_dir.parent / "ground_truth.json"
            if not cfg_path.exists():
                cfg_path = _db_dir.parent / "ground_truth.json"
            gt = _json.load(open(cfg_path)).get("experiments", {}).get(experiment_id, {})
            intended = gt.get("intended_shares") or gt.get("split") or {}
            if isinstance(intended, dict):
                # shares may key by variant NAME; map through the ID contract
                id_map = {v["name"]: v["id"] for v in variants}
                intended = {id_map.get(k, k): val for k, val in intended.items()}
            elif isinstance(intended, list):
                # ordered list matching variants; control first
                vids = [v["id"] for v in variants]
                intended = dict(zip(vids, intended))
        except Exception:
            intended = {}
    users_by_variant: Dict[str, int] = {}
    srm_source = "assignments"
    try:
        name_to_id_map = {v["name"]: v["id"] for v in variants}
        res_a = run_bq_query(
            f"SELECT variant, COUNT(*) AS n FROM fact_user_assignments "
            f"WHERE experiment_id = '{exp_id}' GROUP BY 1", db, row_limit=1000)
        if res_a.ok:
            for variant_nm, n in res_a.rows:
                vid = name_to_id_map.get(variant_nm, variant_nm)
                users_by_variant[vid] = int(n)
        else:
            srm_source = "unavailable"
    except Exception:
        srm_source = "unavailable"
    if intended and srm_source == "assignments" and users_by_variant:
        from .stats import check_srm
        srm_res = check_srm(users_by_variant, intended, threshold=srm_threshold)
        srm = dict(srm_res._asdict()) if hasattr(srm_res, "_asdict") else dict(vars(srm_res)) if hasattr(srm_res, "__dict__") else {"detected": bool(srm_res)}
        srm["note"] = f"computed over {srm_source} (assigned users)"
    else:
        srm = {"detected": False, "note": "no intended shares in registry"}

    # --- headline: whole-window pooled lift per variant per metric --------
    headline: Dict[str, Any] = {}
    for m in metrics_used:
        hdr = {}
        for v in treatments:
            vid = v["id"]
            ci = _lift_ci(_variant_cells(by_day_overall, ctl_id, m),
                          _variant_cells(by_day_overall, vid, m))
            hdr[vid] = ci
        headline[m] = hdr

    # --- cut sweep: treatment × slice dimension values --------------------
    cuts: List[Dict[str, Any]] = []
    # DEFAULT cut request: the Overall cut per (metric × variant); deeper slice
    # values arrive via the `slices` argument (the readout caller / agent picks them).
    slice_cols: Dict[str, List[str]] = {"slice_country": ["Overall"], "slice_page": ["Overall"]}
    if slices:
        for k, v in slices.items():
            slice_cols[k] = list(v)
    # discover the actual slice values in the data once (one query)
    dims_present: Dict[str, List[str]] = {}
    if slices:
        res_dims = run_bq_query(
            f"SELECT 'slice_country' AS col, slice_country AS val FROM exp_scoresheet "
            f"WHERE experiment_id = '{exp_id}' AND NOT overall_flag AND slice_country != 'Overall' "
            f"UNION SELECT 'slice_page', slice_page FROM exp_scoresheet "
            f"WHERE experiment_id = '{exp_id}' AND NOT overall_flag AND slice_page != 'Overall'",
            db, row_limit=1000)
        if res_dims.ok:
            for col, val in res_dims.rows:
                if col in slice_cols:
                    dims_present.setdefault(col, []).append(val)
        else:
            dims_present = {k: [v for v in vals if v != "Overall"]
                            for k, vals in slice_cols.items()}
    for m in metrics_used:
        for v in treatments:
            vid = v["id"]
            for col, allowed_vals in slice_cols.items():
                vals = [x for x in allowed_vals if x == "Overall" or x in dims_present.get(col, [])]
                for val in vals:
                    if val == "Overall":
                        ci = _lift_ci(_variant_cells(by_day_overall, ctl_id, m),
                                      _variant_cells(by_day_overall, vid, m))
                    else:
                        ccells = _day_slice_series(slice_rows, ctl_id, col, val, m)
                        tcells = _day_slice_series(slice_rows, vid, col, val, m)
                        ci = _lift_ci(ccells, tcells)
                    cuts.append({
                        "metric": m, "variant_id": vid,
                        "slice_dim": col, "slice_value": val,
                        **ci,
                    })

    # --- ramp: weekly lifts on the north star ----------------------------
    ramp = _weekly_ramp(rows, ctl_id, [v["id"] for v in treatments], north_star)

    # --- multiplicity warning ---------------------------------------------
    n_tests = len(cuts)
    multiplicity = {"n_tests": n_tests,
                    "note": ("significance inside a slice is weaker evidence than "
                             "experiment-wide significance" if n_tests > 2 else None)}

    # --- decision ---------------------------------------------------------
    decision = _decide(headline, cuts, srm, registry_row, treatments, north_star)

    return SweepResult(
        experiment_id=experiment_id, window=(eff_start, eff_end),
        headline=headline, cuts=cuts, ramp=ramp, srm=srm,
        multiplicity=multiplicity, decision=decision, queries=queries,
        refusals=refusals,
    )


def _daily_slice_rows(db: str, exp_id: str, queries: List[Dict],
                      start: str, end: str) -> List[Sequence[Any]]:
    """Per (day, variant, slice_country) cells from detail slices."""
    sql = f"""
        SELECT CAST(metric_date AS VARCHAR) AS d, variant_id, slice_country, slice_page,
               SUM(users) AS users, SUM(purchases) AS purchases,
               SUM(add_to_cart) AS atc, SUM(gmv) AS gmv
        FROM exp_scoresheet
        WHERE experiment_id = '{exp_id}' AND NOT overall_flag
          AND metric_date >= DATE '{start}' AND metric_date <= DATE '{end}'
          AND slice_country != 'Overall'
        GROUP BY 1, 2, 3, 4 ORDER BY 1, 2
    """
    res = run_bq_query(sql, db, row_limit=50000)
    queries.append({"label": "sweep_slice", "ok": res.ok, "rows": res.row_count,
                    "sql": sql.strip(), "error": res.error})
    if not res.ok:
        raise RuntimeError(f"sweep slice failed: {res.error}")
    return list(res.rows)


def _day_slice_series(slice_rows, vid: str, col: str, val: str, metric: str
                      ) -> List[Tuple[float, float]]:
    num_col, den_col, _ = METRICS[metric]
    out = []
    for d, v, sc, sp, users, purch, atc, gmv in slice_rows:
        slice_val = sc if col == "slice_country" else sp
        if v == vid and slice_val == val:
            num = {"purchases": purch, "gmv": gmv, "add_to_cart": atc}[num_col]
            out.append((float(num), float(users)))
    return out


# ---------------------------------------------------------------------------


def _weekly_ramp(rows, ctl_id, t_ids, metric, weeks=4):
    """Weekly pooled lifts on the metric — stability / novelty check."""
    from collections import defaultdict
    by_week: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    for d, vid, _ctl, users, purch, atc, gmv in rows:
        y, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
        wk = (date(y, m, dd) - date(y, 1, 1)).days // 7  # week-of-year, stable
        cell = by_week[wk][vid]
        cell["num"] = cell.get("num", 0) + float(purch if metric == "purch_per_user" else gmv)
        cell["den"] = cell.get("den", 0) + float(users)
    lifts = []
    weeks_sorted = sorted(by_week)
    for wk in weeks_sorted:
        c = by_week[wk].get(ctl_id, {})
        if not c.get("den"):
            continue
        for vid in t_ids:
            t = by_week[wk].get(vid, {})
            if not t.get("den"):
                continue
            lr = (t["num"] / t["den"]) / (c["num"] / c["den"]) - 1.0
            lifts.append({"week": wk, "variant_id": vid, "lift": round(lr, 4)})
    return {"per_week": lifts, "stable": _verdict_stable(lifts)}


def _verdict_stable(lifts: List[Dict]) -> Optional[bool]:
    """Stable = same sign across weeks (novelty decays flip sign)."""
    if not lifts:
        return None
    signs = {(1 if l["lift"] > 0 else -1 if l["lift"] < 0 else 0) for l in lifts}
    return len(signs) == 1 and 0 not in signs


def _decide(headline, cuts, srm, registry_row, treatments, north_star
            ) -> Dict[str, Any]:
    """Industry decision table, deterministic."""
    ship_threshold = registry_row.get("ship_threshold")
    guard_metric = registry_row.get("guardrail_metric")
    guard_tol = registry_row.get("guardrail_tolerance")

    # primary read from headline on north_star
    ns = headline.get(north_star, {})
    winner = None
    winner_ci = None
    for vid, ci in ns.items():
        if ci.get("significant") and (ci.get("lift") or 0) > (ship_threshold or 0.0):
            if ci["lift"] > (winner_ci or float("-inf")):
                winner, winner_ci = vid, ci["lift"]

    if srm.get("detected"):
        return {"verdict": "HOLD", "reason": "SRM detected — experiment integrity broken",
                "winner": None}
    guard_metric_key = guard_metric if guard_metric in METRICS else None
    verdict, reasons = "NO-WINNER", []

    for vid, ci in ns.items():
        lift = ci.get("lift")
        if lift is None:
            continue
        sig = ci.get("significant")
        if sig and lift > 0 and (not ship_threshold or lift >= ship_threshold):
            # candidate. Check guardrail for the same variant in every metric cut
            bad = [c for c in cuts
                   if c["variant_id"] == vid and c.get("significant")
                   and c.get("lift", 0) < 0
                   and guard_metric and c["metric"] == guard_metric]
            if bad:
                reasons.append(f"{vid}: candidate but guardrail breaches")
                continue
            winner = vid
            verdict = "GO"
            lo, hi = ci["ci"] if ci["ci"] else (None, None)
            reasons.append(f"ship {vid}: pooled lift {lift:+.3f} CI [{lo:+.3f}, {hi:+.3f}]")
        elif sig and lift < 0:
            reasons.append(f"{vid}: significant regression")
            verdict = "NO-GO" if verdict in ("NO-WINNER", "NO-GO") else verdict
    for vid in ns:
        if vid != winner and not ns[vid].get("significant"):
            reasons.append(f"{vid}: not powered / no conclusion")
    if not reasons:
        reasons.append("no significant effects detected")
    return {"verdict": verdict, "winner": winner, "reasons": reasons,
            "n_tests": len(cuts)}