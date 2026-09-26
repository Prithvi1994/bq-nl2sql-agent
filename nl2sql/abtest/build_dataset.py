"""Generate a synthetic-but-statistically-honest A/B testing dataset.

The data is synthetic, but the statistics are exact: every per-user total is drawn once
from a distribution whose mean is the intended metric value, then spread across that
user's active days so the parts sum back to the total. The aggregate a readout recovers
from the daily grain is therefore the aggregate that was generated, and
`ground_truth.json` is a true oracle rather than an approximation.

Three experiments, each a different readout scenario:

  checkout_flow_v2      B genuinely wins on conversion; C regresses a revenue guardrail
  pricing_page_copy_v3  no true effect, but the sample reads positive -> tests restraint
  onboarding_email_v1   genuinely positive, with a 60/40 split -> tests SRM detection

Design rules enforced here:
  * Purchase is modelled per assigned user ("did this user ever purchase"), and every
    assignment gets a row in the outcome views, so the purchase_rate denominator is
    assigned users. Selecting on activity biases lift upward.
  * `active_rate` is low enough that days_active has real spread, so CUPED has signal.
  * AOV is a parameter; revenue_per_user is derived as AOV * purchase_rate, never invented.
  * Treatment effects apply to the test phase only, so the pre phase is a clean baseline.

Usage:
    python -m nl2sql.abtest.build_dataset --out ./abtest_data
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import numpy as np

SEED = 20260925


def _variant_id(name: str, idx: int) -> str:
    """Stable arm keys per the platform contract: control -> ctl, others -> t1/t2."""
    if name == "control":
        return "ctl"
    return f"t{idx}"  # treatment_b -> t1, treatment_c -> t2



COUNTRIES = {
    "US": 0.34, "GB": 0.16, "DE": 0.14, "FR": 0.11, "JP": 0.09, "BR": 0.09, "IN": 0.07,
}
COUNTRY_AOV = {"US": 1.00, "GB": 0.92, "DE": 1.08, "FR": 0.86, "JP": 1.14, "BR": 0.78, "IN": 0.71}
PLATFORMS = {"web": 0.54, "ios": 0.28, "android": 0.18}
PLATFORM_FREQ = {"web": 1.00, "ios": 1.22, "android": 1.15}


@dataclass
class VariantSpec:
    name: str
    share: float
    true_purchase_lift: float = 0.0
    """Relative lift in per-user purchase rate vs control. 0.08 == +8%."""
    true_aov_lift: float = 0.0
    """Relative lift in average order value. Revenue lift is the product of the two."""
    true_sessions_lift: float = 0.0
    true_active_lift: float = 0.0
    true_cart_lift: float = 0.0


@dataclass
class ExperimentSpec:
    experiment_id: str
    name: str
    hypothesis: str
    primary_metric: str
    ship_threshold: float
    guardrail_metric: str
    guardrail_tolerance: float
    intended_shares: Dict[str, float]
    variants: List[VariantSpec]
    n_users: int
    n_days: int
    pre_days: int
    base_purchase_rate: float
    base_aov: float
    sessions_per_active_day: float
    cart_rate: float
    active_rate: float
    day_of_week_effect: float = 0.18
    notes: str = ""


EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec(
        experiment_id="checkout_flow_v2",
        name="One-page checkout",
        hypothesis=(
            "Reducing checkout to a single page lifts purchase conversion by at least 8% "
            "relative to the current three-step flow, without regressing revenue per user."
        ),
        primary_metric="purchase_rate",
        ship_threshold=0.05,
        guardrail_metric="revenue_per_user",
        guardrail_tolerance=-0.02,
        intended_shares={"control": 0.50, "treatment_b": 0.25, "treatment_c": 0.25},
        variants=[
            VariantSpec("control", 0.50),
            VariantSpec("treatment_b", 0.25, 0.15, 0.09, 0.04, 0.05, 0.06),
            VariantSpec("treatment_c", 0.25, 0.02, -0.14, -0.03, 0.01, -0.02),
        ],
        n_users=30000,
        n_days=56,
        pre_days=28,
        base_purchase_rate=0.120,
        base_aov=94.0,
        sessions_per_active_day=2.4,
        cart_rate=0.11,
        active_rate=0.18,
        notes=(
            "B is the intended winner: +15% purchase rate and +9% AOV. C keeps conversion "
            "roughly flat but cuts AOV 14% because it drops expedited shipping, so its "
            "revenue_per_user regresses about 12%. A conversion-only readout ships C."
        ),
    ),
    ExperimentSpec(
        experiment_id="pricing_page_copy_v3",
        name="Pricing page headline rewrite",
        hypothesis=(
            "A benefit-led headline on the pricing page lifts purchase conversion by 3% "
            "or more relative to the current feature-led headline."
        ),
        primary_metric="purchase_rate",
        ship_threshold=0.03,
        guardrail_metric="revenue_per_user",
        guardrail_tolerance=-0.01,
        intended_shares={"control": 0.50, "treatment_b": 0.50},
        variants=[
            VariantSpec("control", 0.50),
            # No treatment effect at all. The pre-period covariate is shifted for
            # treatment_b below, so the naive aggregate reads positive.
            VariantSpec("treatment_b", 0.50),
        ],
        n_users=30000,
        n_days=42,
        pre_days=21,
        base_purchase_rate=0.130,
        base_aov=88.0,
        sessions_per_active_day=1.9,
        cart_rate=0.10,
        active_rate=0.17,
        day_of_week_effect=0.11,
        notes=(
            "True effect is exactly zero on every metric. Treatment users are drawn from a "
            "slightly higher-propensity population, so the unadjusted lift reads about +3% "
            "and clears the threshold. After pre-period adjustment the lift collapses to "
            "zero. A correct readout must not ship this."
        ),
    ),
    ExperimentSpec(
        experiment_id="onboarding_email_v1",
        name="Day-zero onboarding email",
        hypothesis=(
            "A day-zero onboarding email lifts purchase conversion by 10% or more "
            "relative to no email."
        ),
        primary_metric="purchase_rate",
        ship_threshold=0.05,
        guardrail_metric="revenue_per_user",
        guardrail_tolerance=-0.02,
        intended_shares={"control": 0.50, "treatment_b": 0.50},
        variants=[
            VariantSpec("control", 0.40),
            # +9% sessions and +8% active days, not zero: the email drives return
            # visits. sessions_per_user is derived from days_active, so setting only
            # the active lift is enough, but both are declared for clarity.
            VariantSpec("treatment_b", 0.60, 0.14, 0.12, 0.09, 0.08, 0.08),
        ],
        n_users=30000,
        n_days=35,
        pre_days=21,
        base_purchase_rate=0.105,
        base_aov=76.0,
        sessions_per_active_day=1.7,
        cart_rate=0.09,
        active_rate=0.16,
        day_of_week_effect=0.22,
        notes=(
            "The effect is real and large. Assignment is 60/40 rather than the intended "
            "50/50, which is a sample ratio mismatch: roughly a fifth of control traffic "
            "never reached the logging layer. The effect estimate stays correct in "
            "direction, but no ship decision should be made on data whose randomisation "
            "failed this check."
        ),
    ),
]


def _weighted_draw(rng: np.random.Generator, options: Dict[str, float]) -> str:
    keys = list(options)
    probs = np.array([options[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def _split_totals(rng: np.random.Generator, totals: np.ndarray, buckets: np.ndarray) -> List[List[float]]:
    """Split each total into `bucket` parts that sum exactly to the total.

    A weighted multinomial draw, floored, with the residual pushed onto the largest
    fractional parts. This is what lets a readout aggregate the daily table and recover
    the per-user value that was generated.
    """
    out: List[List[float]] = []
    for total, nb in zip(totals, buckets):
        total = float(total)
        nb = int(nb)
        if nb <= 0:
            out.append([])
            continue
        if total <= 0:
            out.append([0.0] * nb)
            continue
        if total < nb:
            parts = [0.0] * nb
            idx = rng.choice(nb, size=int(total), replace=False)
            for i in idx:
                parts[int(i)] = 1.0
            out.append(parts)
            continue
        w = rng.dirichlet(np.full(nb, 2.2))
        raw = w * total
        parts = np.floor(raw)
        residual = total - float(parts.sum())
        order = np.argsort(-(raw - parts))
        for k in range(int(round(residual))):
            parts[order[k % nb]] += 1.0
        out.append([float(x) for x in parts])
    return out


def _build_one(exp: ExperimentSpec, rng: np.random.Generator) -> Dict[str, Any]:
    n = exp.n_users
    start = date(2026, 1, 5)
    pre_start = start - timedelta(days=exp.pre_days)

    countries = np.array([_weighted_draw(rng, COUNTRIES) for _ in range(n)])
    platforms = np.array([_weighted_draw(rng, PLATFORMS) for _ in range(n)])
    country_aov = np.array([COUNTRY_AOV[c] for c in countries])
    platform_freq = np.array([PLATFORM_FREQ[p] for p in platforms])

    # Latent engagement drives activity, sessions and purchase propensity together.
    # This correlation is what makes unadjusted lift misleading and CUPED worthwhile.
    latent = rng.normal(0.0, 0.45, n)
    propensity = np.clip(0.90 + 0.14 * latent, 0.30, 2.20)

    variant_names = [v.name for v in exp.variants]
    shares = np.array([v.share for v in exp.variants], dtype=float)
    shares /= shares.sum()
    variant_idx = rng.choice(len(exp.variants), size=n, p=shares)
    variants = np.array([variant_names[i] for i in variant_idx])

    by_name = {v.name: v for v in exp.variants}
    lift_purchase = np.array([by_name[v].true_purchase_lift for v in variants])
    lift_aov = np.array([by_name[v].true_aov_lift for v in variants])
    lift_sessions = np.array([by_name[v].true_sessions_lift for v in variants])
    lift_active = np.array([by_name[v].true_active_lift for v in variants])
    lift_cart = np.array([by_name[v].true_cart_lift for v in variants])

    # Scenario hook: pricing_page_copy_v3 gives treatment a higher-propensity population
    # with no treatment effect, which is what makes the naive lift look positive.
    if exp.experiment_id == "pricing_page_copy_v3":
        propensity = np.clip(np.where(variants == "treatment_b", propensity * 1.055, propensity), 0.30, 2.20)

    assignments = [
        (
            f"u_{i:06d}",
            exp.experiment_id,
            variants[i],
            (start + timedelta(days=int(rng.integers(0, exp.pre_days)))).isoformat(),
            countries[i],
            platforms[i],
        )
        for i in range(n)
    ]

    metrics: List[tuple] = []

    def emit(phase: str, p_start: date, n_days: int, *, apply_lift: bool) -> None:
        zero = np.zeros(n)
        l_act = lift_active if apply_lift else zero
        l_ses = lift_sessions if apply_lift else zero
        l_cart = lift_cart if apply_lift else zero
        l_pur = lift_purchase if apply_lift else zero
        l_aov = lift_aov if apply_lift else zero

        p_active = np.clip(exp.active_rate * propensity * (1 + l_act), 0.02, 0.92)
        days_active = rng.binomial(n_days, p_active)

        lam = exp.sessions_per_active_day * days_active * propensity * platform_freq * (1 + l_ses)
        sessions = rng.poisson(np.clip(lam, 0, 4000))

        carts = rng.binomial(
            np.clip(sessions, 0, 500), np.clip(exp.cart_rate * (1 + l_cart), 0.001, 0.9)
        )

        # Purchase is an event for the assigned user, not a per-day coin flip.
        p_purchase = np.clip(exp.base_purchase_rate * propensity * (1 + l_pur), 0.001, 0.85)
        purchased = rng.random(n) < p_purchase
        purchase_counts = np.where(
            purchased, 1 + rng.poisson(np.clip(0.16 * p_purchase, 0, 2.0), n), 0
        ).astype(int)

        aov = rng.lognormal(
            mean=np.log(exp.base_aov * country_aov * (1 + l_aov)) - 0.5 * 0.30**2,
            sigma=0.30,
            size=n,
        )
        dur_mu = np.clip(1.6 * sessions, 0, 6000) / np.maximum(days_active, 1)
        duration_total = rng.gamma(3.0, np.clip(dur_mu * 0.32, 0.05, 900)) * days_active

        s_parts = _split_totals(rng, sessions, days_active)
        c_parts = _split_totals(rng, carts, days_active)
        d_parts = _split_totals(rng, duration_total, days_active)

        for i in range(n):
            nb = int(days_active[i])
            if nb == 0:
                continue
            active_dates = sorted(rng.choice(n_days, size=nb, replace=False).tolist())
            remaining = int(purchase_counts[i])
            revenue_remaining = round(purchase_counts[i] * float(aov[i]), 2)
            for j, off in enumerate(active_dates):
                day = p_start + timedelta(days=off)
                if remaining > 0:
                    last = j == len(active_dates) - 1
                    take = min(remaining, 1 if (last or rng.random() < 0.34) else 0)
                    if take:
                        rev_day = revenue_remaining if last else min(
                            round(revenue_remaining / remaining, 2), revenue_remaining
                        )
                        revenue_remaining = round(revenue_remaining - rev_day, 2)
                    else:
                        rev_day = 0.0
                    remaining -= take
                else:
                    take, rev_day = 0, 0.0
                metrics.append(
                    (
                        f"u_{i:06d}",
                        exp.experiment_id,
                        day.isoformat(),
                        phase,
                        int(round(s_parts[i][j])),
                        int(round(s_parts[i][j] * 3.4 * propensity[i])),
                        int(round(c_parts[i][j])),
                        int(take),
                        float(rev_day),
                        round(float(d_parts[i][j]), 1),
                    )
                )

    emit("pre", pre_start, exp.pre_days, apply_lift=False)
    emit("test", start, exp.n_days, apply_lift=True)

    ground_truth = {
        "experiment_id": exp.experiment_id,
        "primary_metric": exp.primary_metric,
        "ship_threshold": exp.ship_threshold,
        "guardrail_metric": exp.guardrail_metric,
        "guardrail_tolerance": exp.guardrail_tolerance,
        "intended_shares": exp.intended_shares,
        "n_users": n,
        "n_days": exp.n_days,
        "pre_days": exp.pre_days,
        "started_on": None,   # filled by build() once dates are fixed
        "ended_on": None,
        "variants": [          # ordered, control first (is_control contract)
            {"id": ("ctl" if v.name == "control" else ("t1" if i == 1 else "t2")), "name": v.name}
            for i, v in enumerate(exp.variants)
        ],
        "true_lift_vs_control": {
            v.name: {
                "purchase_rate": v.true_purchase_lift,
                "aov": v.true_aov_lift,
                "revenue_per_user": round((1 + v.true_purchase_lift) * (1 + v.true_aov_lift) - 1, 6),
                "sessions_per_user": v.true_sessions_lift,
                "days_active": v.true_active_lift,
                "add_to_cart_rate": v.true_cart_lift,
            }
            for v in exp.variants
        },
        "control_baseline": {
            "purchase_rate": exp.base_purchase_rate,
            "aov": exp.base_aov,
            "revenue_per_user": round(exp.base_purchase_rate * exp.base_aov, 4),
        },
        "scenario": exp.notes,
    }
    return {"experiment": exp, "assignments": assignments, "metrics": metrics, "ground_truth": ground_truth}


_COLUMNS = {
    "fact_user_assignments": ["user_id", "experiment_id", "variant", "assigned_on", "country", "platform"],
    "fact_daily_user_metrics": [
        "user_id", "experiment_id", "metric_date", "phase", "sessions", "pageviews",
        "add_to_cart", "purchases", "revenue_usd", "session_duration_s",
    ],
}

_pyarrow_cache: Any = None


def _pyarrow() -> Any:
    global _pyarrow_cache
    if _pyarrow_cache is None:
        import pyarrow as pa

        _pyarrow_cache = pa
    return _pyarrow_cache


DDL = """
CREATE TABLE dim_experiments (
    experiment_id            VARCHAR NOT NULL,
    name                     VARCHAR NOT NULL,
    hypothesis               VARCHAR NOT NULL,
    primary_metric           VARCHAR NOT NULL,
    ship_threshold           DOUBLE  NOT NULL,
    guardrail_metric         VARCHAR NOT NULL,
    guardrail_tolerance      DOUBLE  NOT NULL,
    started_on               DATE    NOT NULL,
    ended_on                 DATE    NOT NULL,
    target_split             VARCHAR NOT NULL
);

CREATE TABLE fact_user_assignments (
    user_id         VARCHAR NOT NULL,
    experiment_id   VARCHAR NOT NULL,
    variant         VARCHAR NOT NULL,
    assigned_on     DATE    NOT NULL,
    country         VARCHAR NOT NULL,
    platform        VARCHAR NOT NULL
);

CREATE TABLE fact_daily_user_metrics (
    user_id           VARCHAR NOT NULL,
    experiment_id     VARCHAR NOT NULL,
    metric_date       DATE    NOT NULL,
    phase             VARCHAR NOT NULL,
    sessions          BIGINT  NOT NULL,
    pageviews         BIGINT  NOT NULL,
    add_to_cart       BIGINT  NOT NULL,
    purchases         BIGINT  NOT NULL,
    revenue_usd       DOUBLE  NOT NULL,
    session_duration_s DOUBLE NOT NULL
);
"""

VIEWS = [
    # Daily grain with the variant attached. variant lives in the assignment table, so
    # every view joins assignments explicitly rather than relying on a denormalised column.
    ("fact_daily_variant_metrics",
     "SELECT m.metric_date, m.experiment_id, a.variant, m.phase, "
     "COUNT(DISTINCT m.user_id) AS users, SUM(m.sessions) AS sessions, "
     "SUM(m.pageviews) AS pageviews, SUM(m.add_to_cart) AS add_to_cart, "
     "SUM(m.purchases) AS purchases, SUM(m.revenue_usd) AS revenue_usd "
     "FROM fact_daily_user_metrics m JOIN fact_user_assignments a USING (user_id, experiment_id) "
     "GROUP BY 1,2,3,4"),
    ("fact_user_experiment_totals",
     "SELECT m.user_id, m.experiment_id, a.variant, MIN(m.metric_date) AS first_date, "
     "COUNT(DISTINCT m.metric_date) AS days_active, SUM(m.sessions) AS sessions, "
     "SUM(m.pageviews) AS pageviews, SUM(m.add_to_cart) AS add_to_cart, "
     "SUM(m.purchases) AS purchases, SUM(m.revenue_usd) AS revenue_usd "
     "FROM fact_daily_user_metrics m JOIN fact_user_assignments a USING (user_id, experiment_id) "
     "GROUP BY 1,2,3"),
    ("fact_pre_period_totals",
     "SELECT m.user_id, m.experiment_id, a.variant, SUM(m.sessions) AS pre_sessions, "
     "SUM(m.pageviews) AS pre_pageviews, SUM(m.revenue_usd) AS pre_revenue, "
     "SUM(m.add_to_cart) AS pre_add_to_cart, SUM(m.purchases) AS pre_purchases, "
     "SUM(m.session_duration_s) AS pre_session_duration_s, "
     "COUNT(DISTINCT m.metric_date) AS pre_days "
     "FROM fact_daily_user_metrics m JOIN fact_user_assignments a USING (user_id, experiment_id) "
     "WHERE m.phase = 'pre' GROUP BY 1,2,3"),
    ("fact_test_phase_user_totals",
     "SELECT m.user_id, m.experiment_id, a.variant, COUNT(DISTINCT m.metric_date) AS days_active, "
     "SUM(m.sessions) AS sessions, SUM(m.pageviews) AS pageviews, SUM(m.add_to_cart) AS add_to_cart, "
     "SUM(m.purchases) AS purchases, SUM(m.revenue_usd) AS revenue_usd "
     "FROM fact_daily_user_metrics m JOIN fact_user_assignments a USING (user_id, experiment_id) "
     "WHERE m.phase = 'test' GROUP BY 1,2,3"),
    # Assigned-user grain: every assignment has a row even with no activity, so the
    # denominator of purchase_rate never depends on who happened to show up.
    ("fact_assigned_user_outcomes",
     "SELECT a.user_id, a.experiment_id, a.variant, a.country, a.platform, "
     "COALESCE(t.days_active, 0) AS days_active, COALESCE(t.sessions, 0) AS sessions, "
     "COALESCE(t.pageviews, 0) AS pageviews, COALESCE(t.add_to_cart, 0) AS add_to_cart, "
     "COALESCE(t.purchases, 0) AS purchases, COALESCE(t.revenue_usd, 0.0) AS revenue_usd "
     "FROM fact_user_assignments a "
     "LEFT JOIN fact_test_phase_user_totals t USING (user_id, experiment_id)"),
    ("fact_daily_assigned",
     "SELECT m.metric_date, m.user_id, m.experiment_id, a.variant, a.country, a.platform, "
     "m.phase, m.sessions, m.pageviews, m.add_to_cart, m.purchases, m.revenue_usd, "
     "m.session_duration_s "
     "FROM fact_daily_user_metrics m JOIN fact_user_assignments a USING (user_id, experiment_id)"),
    # Windowed grain: one row per assigned user per test day, zero-filled. This is what
    # the MDL queries for a date range, because experiment_user above is pre-aggregated
    # over the whole test phase and therefore cannot honour a requested window. A user
    # with no activity has no rows here, so the GROUP BY in the query is what preserves
    # them in the denominator.
    ("fact_assigned_daily",
     "SELECT a.user_id, a.experiment_id, a.variant, a.country, a.platform, "
     "m.metric_date, "
     "COALESCE(m.sessions, 0) AS sessions, "
     "COALESCE(m.pageviews, 0) AS pageviews, COALESCE(m.add_to_cart, 0) AS add_to_cart, "
     "COALESCE(m.purchases, 0) AS purchases, COALESCE(m.revenue_usd, 0.0) AS revenue_usd, "
     "COALESCE(m.session_duration_s, 0.0) AS session_duration_s "
     "FROM fact_user_assignments a "
     "LEFT JOIN fact_daily_user_metrics m "
     "  ON m.user_id = a.user_id AND m.experiment_id = a.experiment_id "
     "WHERE m.phase = 'test' OR m.metric_date IS NULL"),
    ("fact_assigned_with_pre",
     "SELECT a.user_id, a.experiment_id, a.variant, a.country, a.platform, "
     "COALESCE(t.days_active, 0) AS days_active, COALESCE(t.sessions, 0) AS sessions, "
     "COALESCE(t.pageviews, 0) AS pageviews, COALESCE(t.add_to_cart, 0) AS add_to_cart, "
     "COALESCE(t.purchases, 0) AS purchases, COALESCE(t.revenue_usd, 0.0) AS revenue_usd, "
     "COALESCE(p.pre_sessions, 0) AS pre_sessions, COALESCE(p.pre_pageviews, 0) AS pre_pageviews, "
     "COALESCE(p.pre_revenue, 0.0) AS pre_revenue, COALESCE(p.pre_days, 0) AS pre_days, "
     "COALESCE(p.pre_add_to_cart, 0) AS pre_add_to_cart, COALESCE(p.pre_purchases, 0) AS pre_purchases, "
     "COALESCE(p.pre_session_duration_s, 0.0) AS pre_session_duration_s "
     "FROM fact_user_assignments a "
     "LEFT JOIN fact_test_phase_user_totals t USING (user_id, experiment_id) "
     "LEFT JOIN fact_pre_period_totals p USING (user_id, experiment_id)"),
]


def build(out_dir: str, scale: float = 1.0) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "abtest.duckdb"
    if db_path.exists():
        db_path.unlink()
    truth_path = out / "ground_truth.json"

    rng = np.random.default_rng(SEED)
    conn = duckdb.connect(str(db_path))
    conn.execute(DDL)

    def _insert(table: str, rows: List[tuple]) -> int:
        """Bulk insert through Arrow; row-at-a-time executemany is orders of magnitude slower."""
        if not rows:
            return 0
        pa = _pyarrow()
        cols = _COLUMNS[table]
        tbl = pa.Table.from_pylist([dict(zip(cols, r)) for r in rows])
        conn.register(f"_stage_{table}", tbl)
        joined = ", ".join(cols)
        conn.execute(f"INSERT INTO {table} ({joined}) SELECT {joined} FROM _stage_{table}")
        conn.unregister(f"_stage_{table}")
        return len(rows)

    all_truth: Dict[str, Any] = {}
    summary: List[Dict[str, Any]] = []
    for exp in EXPERIMENTS:
        if scale != 1.0:
            # asdict(exp) recursively converts VariantSpec objects into dicts;
            # rebuilding with ** would hand _build_one dicts where it expects
            # VariantSpec. Only the scalar fields are scaled — keep everything else.
            scalar = {k: v for k, v in asdict(exp).items() if k != "variants"}
            exp = ExperimentSpec(
                variants=exp.variants,
                **{**scalar, "n_users": max(400, int(exp.n_users * scale))},
            )
        built = _build_one(exp, rng)
        e: ExperimentSpec = built["experiment"]

        start = date(2026, 1, 5)
        end = start + timedelta(days=e.n_days - 1)
        target = "/".join(f"{k}:{v:.0%}" for k, v in e.intended_shares.items())
        conn.execute(
            "INSERT INTO dim_experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
            [e.experiment_id, e.name, e.hypothesis, e.primary_metric, e.ship_threshold,
             e.guardrail_metric, e.guardrail_tolerance, start.isoformat(), end.isoformat(), target],
        )
        _insert("fact_user_assignments", built["assignments"])
        _insert("fact_daily_user_metrics", built["metrics"])
        gt = built["ground_truth"]
        gt["started_on"] = start.isoformat()
        gt["ended_on"] = end.isoformat()
        all_truth[e.experiment_id] = gt
        summary.append(
            {
                "experiment_id": e.experiment_id,
                "users": e.n_users,
                "rows": len(built["metrics"]),
                "intended_shares": e.intended_shares,
                "true_purchase_lift": {v.name: v.true_purchase_lift for v in e.variants},
                "true_revenue_lift": {
                    v.name: round((1 + v.true_purchase_lift) * (1 + v.true_aov_lift) - 1, 4)
                    for v in e.variants
                },
            }
        )

    for name, body in VIEWS:
        conn.execute(f"CREATE VIEW {name} AS {body}")
    _build_scoresheet(conn, all_truth)
    conn.close()

    truth_path.write_text(json.dumps({"seed": SEED, "experiments": all_truth}, indent=2) + "\n", encoding="utf-8")
    return {"db_path": str(db_path), "ground_truth": str(truth_path), "experiments": summary}





# ---------------------------------------------------------------------------
# exp_scoresheet: the production-shaped precomputed-cut layer (17 col contract)
# ---------------------------------------------------------------------------
# One row per (experiment, metric_date, variant_id, slice). Outcome totals are
# event-day values; per-user rates and lifts are PRECOMPUTED by the platform
# (pooled sum-ratio semantics, not recomputable upstream) and marked never-sum.
# Slides: P13N-style slice dims become (country, page); 'Overall' marker rows
# carry the whole-test cut per variant so the platform serves them instantly.

SCORESHEET_DDL = """
CREATE TABLE exp_scoresheet (
    experiment_id   VARCHAR NOT NULL,
    metric_date     DATE,               -- snapshot date (NULL on Overall rows)
    variant_id      VARCHAR NOT NULL,   -- stable arm key (ctl / t1 / t2)
    variant_nm      VARCHAR NOT NULL,   -- display label only
    is_control      INTEGER NOT NULL,   -- 1 = baseline; from config, never inferred
    slice_country   VARCHAR NOT NULL,   -- 'Overall' = all countries
    slice_page      VARCHAR NOT NULL,   -- 'Overall' = all pages
    overall_flag    BOOLEAN NOT NULL,   -- TRUE = whole-test rollup row
    users           BIGINT  NOT NULL,   -- denominator; never-sum (window uncertain)
    purchases       BIGINT  NOT NULL,   -- safe to sum ONLY across disjoint slices
    add_to_cart     BIGINT  NOT NULL,   -- same
    gmv             DOUBLE  NOT NULL,   -- same (decimal in production)
    purch_per_user  DOUBLE  NOT NULL,   -- precomputed pooled rate; never sum
    gmv_per_user    DOUBLE  NOT NULL,   -- the north star, precomputed; never sum
    purch_lift      DOUBLE,             -- vs matching control, fraction; t rows only
    gmv_tot_lift    DOUBLE              -- TRAP: name says total; IS per-user lift
)
"""


def _build_scoresheet(conn, all_truth: Dict[str, Any]) -> None:
    """Materialize the precomputed-cut layer from the daily truth tables."""
    conn.execute(SCORESHEET_DDL)

    for exp_id, truth in all_truth.items():
        started_on = date.fromisoformat(truth["started_on"])
        ended_on = date.fromisoformat(truth["ended_on"])
        variants = truth["variants"]          # ordered, control first
        control_id = variants[0]["id"]

        daily = conn.execute(
            """
            SELECT metric_date, variant, SUM(users) AS users, SUM(purchases) AS purchases,
                   SUM(add_to_cart) AS atc, SUM(revenue_usd) AS gmv
            FROM fact_daily_variant_metrics
            WHERE experiment_id = ?
              AND metric_date BETWEEN ? AND ?
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            [exp_id, truth["started_on"], truth["ended_on"]],
        ).fetchall()

        # daily rows key by variant NAME (view column); truth uses stable IDs.
        # Map name->id once so all lookups go through the ID contract.
        name_to_id = {v["name"]: v["id"] for v in variants}
        by_day: Dict[date, Dict[str, Dict[str, float]]] = {}
        for d, v, u, p, a, g in daily:
            by_day.setdefault(d, {})[name_to_id[v]] = {"users": u, "purchases": p, "atc": a, "gmv": g}

        rows = []

        def _rate(num, den):
            return (num / den) if den else 0.0

        # precompute the whole-test cut first: lifts need the control totals
        tot: Dict[str, Dict[str, float]] = {}
        for v in variants:
            vid = v["id"]
            users = sum(by_day.get(d, {}).get(vid, {}).get("users", 0) for d in by_day)
            purch = sum(by_day.get(d, {}).get(vid, {}).get("purchases", 0) for d in by_day)
            atc = sum(by_day.get(d, {}).get(vid, {}).get("atc", 0) for d in by_day)
            gmv = sum(by_day.get(d, {}).get(vid, {}).get("gmv", 0.0) for d in by_day)
            tot[vid] = {"users": users, "purchases": purch, "atc": atc, "gmv": gmv}

        for v in variants:
            vid = v["id"]
            is_ctl = 1 if vid == control_id else 0
            t = tot[vid]
            rows.append((
                exp_id, None, vid, v["name"], is_ctl,
                "Overall", "Overall", True,
                int(t["users"]), int(t["purchases"]), int(t["atc"]), round(t["gmv"], 2),
                round(_rate(t["purchases"], t["users"]), 6),
                round(_rate(t["gmv"], t["users"]), 6),
                None if is_ctl else round(
                    (t["purchases"] / tot[control_id]["purchases"]) /
                    (t["users"] / tot[control_id]["users"]) - 1.0, 6) if t["users"] else None,
                None if is_ctl else round(
                    (t["gmv"] / t["users"]) / (tot[control_id]["gmv"] / tot[control_id]["users"]) - 1.0, 6)
                if t["users"] else None,
            ))

        # detail rows: per-day, per-variant; lifts stored NOT day-level (platform
        # computes lift per whole cut, not per day) -> detail lift columns NULL
        for d in sorted(by_day):
            for v in variants:
                vid = v["id"]
                cell = by_day[d].get(vid, {"users": 0, "purchases": 0, "atc": 0, "gmv": 0.0})
                rows.append((
                    exp_id, d, vid, v["name"], 1 if vid == control_id else 0,
                    "Overall", "Overall", False,
                    int(cell["users"]), int(cell["purchases"]), int(cell["atc"]),
                    round(cell["gmv"], 2),
                    round(_rate(cell["purchases"], cell["users"]), 6),
                    round(_rate(cell["gmv"], cell["users"]), 6),
                    None, None,
                ))

        # simpler: parameterized insert via arrow
        cols = ["experiment_id", "metric_date", "variant_id", "variant_nm", "is_control",
                "slice_country", "slice_page", "overall_flag", "users", "purchases",
                "add_to_cart", "gmv", "purch_per_user", "gmv_per_user",
                "purch_lift", "gmv_tot_lift"]
        pa = _pyarrow()
        tbl = pa.Table.from_pylist([dict(zip(cols, r)) for r in rows])
        conn.register("_stage_ss", tbl)
        joined = ", ".join(cols)
        conn.execute(f"INSERT INTO exp_scoresheet ({joined}) SELECT {joined} FROM _stage_ss")
        conn.unregister("_stage_ss")



def main() -> None:
    ap = argparse.ArgumentParser(description="Build the offline A/B test dataset.")
    ap.add_argument("--out", default="./abtest_data")
    ap.add_argument("--scale", type=float, default=1.0, help="Shrink user counts for a fast dev loop.")
    args = ap.parse_args()
    print(json.dumps(build(args.out, args.scale), indent=2))


if __name__ == "__main__":
    main()