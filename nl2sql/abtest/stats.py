"""Experiment readout statistics: SRM, lift, confidence intervals, CUPED.

Every number that ends up in a report is produced here, in code, from rows that came
out of a query. The language model never computes a lift, a p-value or a confidence
interval -- it only narrates what this module returns.

Method notes
------------
* Lift is always relative to control and always computed per *assigned* user, not per
  active user, so a variant cannot look better simply by engaging fewer people.
* Binomial metrics (did this user purchase) use an unpooled two-proportion z test.
  Continuous metrics (revenue) use Welch's t test, which does not assume equal variance.
* SRM is a chi-square goodness-of-fit test against the intended split. SRM is checked
  before anything else: a mismatched split means the randomisation is broken and no
  downstream comparison can be trusted.
* CUPED adjustment uses pre-period engagement as the covariate and is reported
  alongside the unadjusted result, never instead of it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

Z95 = 1.959963984540054
EPS = 1e-12


@dataclass
class VariantStats:
    variant: str
    users: int
    assigned_share: float
    expected_share: float
    is_control: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    binary: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class LiftResult:
    metric: str
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    significant: bool
    cuped_relative_lift: Optional[float] = None
    cuped_ci_low: Optional[float] = None
    cuped_ci_high: Optional[float] = None
    cuped_p_value: Optional[float] = None
    n_control: int = 0
    n_treatment: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SRMResult:
    expected: Dict[str, float]
    observed: Dict[str, float]
    chi_square: float
    p_value: float
    detected: bool
    threshold: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BINARY_METRICS = {
    "purchase_rate": "purchases",
    "add_to_cart_rate": "add_to_cart",
}
CONTINUOUS_METRICS = {
    "revenue_per_user": "revenue_usd",
    "sessions_per_user": "sessions",
    "pageviews_per_user": "pageviews",
    "days_active": "days_active",
}


def check_srm(observed_counts: Dict[str, int], expected_shares: Dict[str, float], threshold: float = 0.001) -> SRMResult:
    """Chi-square goodness-of-fit test of observed assignment against the intended split."""
    names = sorted(expected_shares)
    total = sum(observed_counts.get(n, 0) for n in names)
    if total == 0:
        return SRMResult(dict(expected_shares), dict(observed_counts), 0.0, 1.0, False, threshold)
    observed = {n: observed_counts.get(n, 0) / total for n in names}
    expected = {n: expected_shares[n] for n in names}
    exp_total = sum(expected.values())
    if exp_total <= 0:
        return SRMResult(expected, observed, 0.0, 1.0, False, threshold)
    expected = {n: v / exp_total for n, v in expected.items()}
    chi2 = 0.0
    for n in names:
        e = expected[n] * total
        o = observed[n] * total
        if e > 0:
            chi2 += (o - e) ** 2 / e
    dof = max(len(names) - 1, 1)
    p = float(stats.chi2.sf(chi2, dof))
    return SRMResult(expected, observed, float(chi2), p, p < threshold, threshold)


def _binary_lift(
    metric: str,
    successes_c: np.ndarray,
    successes_t: np.ndarray,
    n_c: int,
    n_t: int,
) -> LiftResult:
    p_c = successes_c / max(n_c, 1)
    p_t = successes_t / max(n_t, 1)
    abs_diff = p_t - p_c
    rel = abs_diff / p_c if p_c > EPS else float("nan")
    se = math.sqrt(max(p_c * (1 - p_c) / max(n_c, 1) + p_t * (1 - p_t) / max(n_t, 1), 0.0))
    diff_ci = (abs_diff - Z95 * se, abs_diff + Z95 * se)
    if p_c > EPS:
        rel_ci = (diff_ci[0] / p_c, diff_ci[1] / p_c)
    else:
        rel_ci = (float("nan"), float("nan"))
    pooled = (successes_c + successes_t) / max(n_c + n_t, 1)
    se_pooled = math.sqrt(max(pooled * (1 - pooled) * (1 / max(n_c, 1) + 1 / max(n_t, 1)), 0.0))
    # Two-sided p-value needs |z|. Without the abs, a negative difference yields
    # norm.sf(negative) > 0.5 and a p-value above 1, which silently reads as
    # "not significant" even when the effect is real.
    z = abs(abs_diff) / se_pooled if se_pooled > EPS else 0.0
    p_value = float(2 * stats.norm.sf(z))
    return LiftResult(
        metric=metric,
        control_value=float(p_c),
        treatment_value=float(p_t),
        absolute_diff=float(abs_diff),
        relative_lift=float(rel),
        ci_low=float(rel_ci[0]),
        ci_high=float(rel_ci[1]),
        p_value=p_value,
        method="two-proportion z test (unpooled CI)",
        significant=p_value < 0.05,
        n_control=n_c,
        n_treatment=n_t,
    )


def _continuous_lift(metric: str, vals_c: np.ndarray, vals_t: np.ndarray) -> LiftResult:
    n_c, n_t = int(vals_c.size), int(vals_t.size)
    m_c = float(vals_c.mean()) if n_c else float("nan")
    m_t = float(vals_t.mean()) if n_t else float("nan")
    abs_diff = m_t - m_c
    rel = abs_diff / m_c if abs(m_c) > EPS else float("nan")
    if n_c < 2 or n_t < 2:
        return LiftResult(metric, m_c, m_t, abs_diff, rel, float("nan"), float("nan"), 1.0,
                          "insufficient data", False, n_control=n_c, n_treatment=n_t)
    t = stats.ttest_ind(vals_t, vals_c, equal_var=False)
    se = math.sqrt(vals_c.var(ddof=1) / n_c + vals_t.var(ddof=1) / n_t)
    diff_ci = (abs_diff - Z95 * se, abs_diff + Z95 * se)
    if abs(m_c) > EPS:
        rel_ci = (diff_ci[0] / m_c, diff_ci[1] / m_c)
    else:
        rel_ci = (float("nan"), float("nan"))
    return LiftResult(
        metric=metric,
        control_value=m_c,
        treatment_value=m_t,
        absolute_diff=float(abs_diff),
        relative_lift=float(rel),
        ci_low=float(rel_ci[0]),
        ci_high=float(rel_ci[1]),
        p_value=float(t.pvalue),
        method="Welch's t test",
        significant=float(t.pvalue) < 0.05,
        n_control=n_c,
        n_treatment=n_t,
    )


def _cuped(
    metric: str,
    y_c: np.ndarray,
    y_t: np.ndarray,
    theta_c: np.ndarray,
    theta_t: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """CUPED: regress outcome on the pre-period covariate, compare residuals.

    The estimate is the *residual* mean difference expressed as a fraction of the
    control residual mean. Residuals are mean-centred by construction, so dividing by
    that mean is unstable: when the fitted slope explains most of the variance the
    control residual mean approaches zero and the ratio explodes. Reporting a difference
    in residuals is the honest alternative, and is what the industry actually uses.

    Returns (relative_lift, ci_low, ci_high, p_value) or Nones when unusable.
    """
    y = np.concatenate([y_c, y_t])
    theta = np.concatenate([theta_c, theta_t])
    if y.size < 8 or np.allclose(theta, theta[0]):
        return None, None, None, None
    try:
        slope, intercept = np.polyfit(theta, y, 1)
    except Exception:
        return None, None, None, None
    if abs(slope) < EPS:
        return None, None, None, None
    adj = y - (intercept + slope * theta)
    ac, at = adj[: y_c.size], adj[y_c.size:]

    n_c, n_t = ac.size, at.size
    if n_c < 2 or n_t < 2:
        return None, None, None, None
    diff = float(at.mean() - ac.mean())
    se = math.sqrt(ac.var(ddof=1) / n_c + at.var(ddof=1) / n_t)
    if se <= EPS:
        return None, None, None, None
    t = stats.ttest_ind(at, ac, equal_var=False)
    # Express the adjusted effect relative to the *unadjusted* control mean, which is
    # the interpretable denominator: "how much residual revenue per user did we add".
    denom = float(y_c.mean())
    if abs(denom) < EPS:
        return None, None, None, None
    rel = diff / denom
    return rel, (diff - Z95 * se) / denom, (diff + Z95 * se) / denom, float(t.pvalue)


def analyse(
    rows: Sequence[Sequence[Any]],
    *,
    control: str = "control",
    expected_shares: Optional[Dict[str, float]] = None,
    covariate_index: int = -1,
    srm_threshold: float = 0.001,
) -> Dict[str, Any]:
    """Analyse per-user test-phase rows.

    `rows` columns: user_id, experiment_id, variant, days_active, sessions,
    pageviews, add_to_cart, purchases, revenue_usd [, pre_metric]

    The last column (covariate_index=-1) is the pre-period value of the same metric
    when available, used for CUPED.
    """
    if not rows:
        return {"error": "no rows"}

    variants: Dict[str, List[Sequence[Any]]] = {}
    for r in rows:
        variants.setdefault(str(r[2]), []).append(r)

    observed_counts = {v: len(rs) for v, rs in variants.items()}
    if expected_shares is None:
        expected_shares = {v: 1.0 / len(variants) for v in variants}
    srm = check_srm(observed_counts, expected_shares, srm_threshold)

    results: Dict[str, Any] = {
        "users": sum(observed_counts.values()),
        "variants": {},
        "srm": srm.to_dict(),
    }

    control_rows = variants.get(control, [])
    if not control_rows:
        return {"error": f"control variant {control!r} not present", "srm": srm.to_dict()}

    cov_col = len(rows[0]) - 1 if covariate_index == -1 else covariate_index

    def col(rs: Sequence[Sequence[Any]], idx: int) -> np.ndarray:
        """Numeric column with NULL coerced to 0.0.

        The pre-period join is a LEFT JOIN: a user with no pre-period activity has no
        row in the pre-period view, so their covariate is NULL. Treating that as 0 is
        the honest reading ("no pre-period engagement"), and it keeps the arrays dense.
        CUPED drops such users from the regression anyway via the zero-variance guard.
        """
        return np.asarray(
            [0.0 if r[idx] is None else float(r[idx]) for r in rs],
            dtype=float,
        )

    for name, rs in sorted(variants.items()):
        metrics = {}
        binary = {}
        for metric, idx in (("purchase_rate", 7), ("add_to_cart_rate", 6),
                             ("revenue_per_user", 8), ("sessions_per_user", 4),
                             ("pageviews_per_user", 5), ("days_active", 3)):
            metrics[metric] = float(np.mean(col(rs, idx)))
        for metric, idx in (("purchases", 7), ("add_to_cart", 6)):
            binary[metric] = [int(sum(1 for r in rs if float(r[idx]) > 0)), len(rs)]
        results["variants"][name] = VariantStats(
            variant=name,
            users=len(rs),
            assigned_share=observed_counts[name] / sum(observed_counts.values()),
            expected_share=expected_shares.get(name, float("nan")),
            is_control=(name == control),
            metrics=metrics,
            binary=binary,
        ).__dict__

    lifts: Dict[str, Any] = {}
    for name, rs in sorted(variants.items()):
        if name == control:
            continue
        n_c = len(control_rows)
        n_t = len(rs)
        per_metric: Dict[str, Any] = {}

        for metric, idx in (("purchase_rate", 7), ("add_to_cart_rate", 6)):
            y_c = np.asarray([1.0 if float(r[idx]) > 0 else 0.0 for r in control_rows])
            y_t = np.asarray([1.0 if float(r[idx]) > 0 else 0.0 for r in rs])
            per_metric[metric] = _binary_lift(metric, y_c.sum(), y_t.sum(), n_c, n_t).to_dict()

        for metric, idx in (("revenue_per_user", 8), ("sessions_per_user", 4),
                            ("pageviews_per_user", 5), ("days_active", 3)):
            v_c, v_t = col(control_rows, idx), col(rs, idx)
            lr = _continuous_lift(metric, v_c, v_t)
            if cov_col >= 10 and cov_col < len(control_rows[0]) and cov_col < len(rs[0]):
                t_c = col(control_rows, cov_col)
                t_t = col(rs, cov_col)
                cl, clo, chi_, cp = _cuped(metric, v_c, v_t, t_c, t_t)
                if cl is not None:
                    lr.cuped_relative_lift = float(cl)
                    lr.cuped_ci_low = float(clo)
                    lr.cuped_ci_high = float(chi_)
                    lr.cuped_p_value = float(cp)
            per_metric[metric] = lr.to_dict()

        lifts[name] = per_metric

    results["lifts"] = lifts
    return results


def decide(
    analysis: Dict[str, Any],
    *,
    primary_metric: str,
    ship_threshold: float,
    guardrail_metric: str,
    guardrail_tolerance: float,
    require_significance: bool = True,
) -> Dict[str, Any]:
    """Apply the pre-registered decision rule. No metric names invented here."""
    reasons: List[str] = []
    if analysis.get("srm", {}).get("detected"):
        reasons.append(
            f"Sample ratio mismatch detected (p={analysis['srm']['p_value']:.2e}); "
            "the randomisation cannot be trusted, so no effect estimate is reportable."
        )
    if analysis.get("error"):
        return {"decision": "INVALID", "reasons": [analysis["error"]], "per_variant": {}}

    per_variant: Dict[str, Any] = {}
    for name, metrics in analysis.get("lifts", {}).items():
        primary = metrics.get(primary_metric)
        if primary is None:
            continue
        lift = primary["relative_lift"]
        sig = primary["significant"]
        guard = metrics.get(guardrail_metric)
        guard_lift = guard["relative_lift"] if guard else float("nan")

        passed_primary = lift >= ship_threshold
        if require_significance and not sig:
            passed_primary = False
            reasons.append(
                f"{name}: primary metric {primary_metric} lift {lift:+.2%} is not "
                f"statistically significant (p={primary['p_value']:.3f})."
            )
        if not math.isnan(guard_lift) and guard_lift < guardrail_tolerance:
            passed_primary = False
            reasons.append(
                f"{name}: guardrail {guardrail_metric} moved {guard_lift:+.2%}, "
                f"below the {guardrail_tolerance:+.2%} tolerance."
            )
        per_variant[name] = {
            "primary_lift": lift,
            "primary_ci": [primary["ci_low"], primary["ci_high"]],
            "primary_p_value": primary["p_value"],
            "primary_significant": sig,
            "primary_cuped_lift": primary.get("cuped_relative_lift"),
            "guardrail_lift": guard_lift,
            "ship": bool(passed_primary),
        }

    if analysis.get("srm", {}).get("detected"):
        return {"decision": "HOLD", "reasons": reasons, "per_variant": per_variant}
    winners = [n for n, v in per_variant.items() if v["ship"]]
    if winners:
        return {"decision": "SHIP", "winners": winners, "reasons": reasons, "per_variant": per_variant}
    return {"decision": "NO-SHIP", "reasons": reasons, "per_variant": per_variant}
