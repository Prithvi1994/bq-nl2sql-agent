# Metric catalog

Authoritative definitions. The SQL column is what the MDL measure expands to; it is
shown so a reviewer can check the definition without reading YAML. When these and
`cubes/experiment_metrics/metadata.yml` disagree, the MDL is what executes — fix
the MDL and regenerate this file.

## conversion

| Metric | Definition | Expands to |
|---|---|---|
| `purchase_rate` | Share of **assigned** users with ≥1 purchase | `SAFE_DIVIDE(COUNT(DISTINCT IF(purchases>0,user_id,NULL)), COUNT(*))` |
| `add_to_cart_rate` | Share of **assigned** users who added to cart | `SAFE_DIVIDE(COUNT(DISTINCT IF(add_to_cart>0,user_id,NULL)), COUNT(*))` |
| `purchasing_users` | Distinct users with ≥1 purchase | `COUNT(DISTINCT IF(purchases>0,user_id,NULL))` |
| `add_to_cart_users` | Distinct users who added to cart | `COUNT(DISTINCT IF(add_to_cart>0,user_id,NULL))` |

## revenue

| Metric | Definition | Expands to |
|---|---|---|
| `revenue_per_user` | Revenue ÷ **assigned** users (incl. zero spenders) | `SAFE_DIVIDE(SUM(revenue_usd), COUNT(*))` |
| `revenue_per_purchaser` | AOV: revenue ÷ purchase **events** | `SAFE_DIVIDE(SUM(revenue_usd), SUM(purchases))` |
| `total_revenue` | Total revenue USD | `SUM(revenue_usd)` |
| `total_purchases` | Purchase events, not distinct purchasers | `SUM(purchases)` |

## engagement

| Metric | Definition | Expands to |
|---|---|---|
| `sessions_per_user` | Sessions ÷ **assigned** users | `SAFE_DIVIDE(SUM(sessions), COUNT(*))` |
| `pageviews_per_user` | Pageviews ÷ **assigned** users | `SAFE_DIVIDE(SUM(pageviews), COUNT(*))` |
| `days_active` | Active days summed over users (a count, not a rate) | `SUM(days_active)` |
| `session_duration_s` | Session seconds ÷ **assigned** users | `SAFE_DIVIDE(SUM(session_duration_s), COUNT(*))` |
| `total_sessions` | Total sessions | `SUM(sessions)` |
| `total_pageviews` | Total pageviews | `SUM(pageviews)` |

## population

| Metric | Definition | Expands to |
|---|---|---|
| `assigned_users` | The denominator of record | `COUNT(*)` |
| `active_users` | Users with ≥1 active day — **never a denominator** | `COUNT(DISTINCT user_id)` on `experiment_daily` |

## Statistics that are not metrics

Computed in Python (`stats.py`) from per-user rows, never in SQL, because they need
the distribution rather than an aggregate:

- **SRM** — chi-square on assignment counts vs `dim_experiments.target_split`. A
  detected SRM blocks naming any winner, however significant the effect.
- **Lift + CI** — Wilson for binary rates, Welch's t for continuous.
- **CUPED** — uses `pre_sessions`, `pre_pageviews`, `pre_revenue`, `pre_days` as
  covariates. Expressed as a ratio over the **unadjusted** control mean; residual
  means are mean-centred by construction, so dividing by one produces a ratio that
  diverges.
- **Ship rule** — primary metric CI must exclude zero *and* exceed
  `dim_experiments.ship_threshold` *and* the guardrail must hold within
  `guardrail_tolerance`. All three, or the verdict is HOLD.
