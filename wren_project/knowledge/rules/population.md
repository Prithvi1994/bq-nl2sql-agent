# Population rules (daily-aggregated grain)

These are policies, not schema facts. The MDL/cube encodes what each measure *is*; this
file encodes which measure is legal in which context. An agent that follows the MDL
but not this file will produce a valid, plausible, wrong number.

The dataset is synthetic with no PII. Row-level privacy rules re-enter here when
real user data lands; nothing below assumes their absence.

## 1. The denominator is SUM(users), never COUNT of day-rows

The grain is one row per day per variant. `COUNT(*)` over these rows counts
**user-days**, not users — in a 90-day test that inflates every per-user rate
roughly 90x while looking completely ordinary.

Every rate divides by `assigned_users` = `SUM(users)` pooled over the window.

```sql
-- WRONG: denominator is user-days
SELECT SAFE_DIVIDE(SUM(purchases), COUNT(*)) FROM experiment_daily
-- RIGHT: pooled sum-ratio
SELECT SAFE_DIVIDE(SUM(purchases), SUM(users)) FROM experiment_daily
```

## 2. Rates are pooled sum-ratios, never averages of day rates

`SUM(numerator) / SUM(users)`. `AVG(day_rate)` weights a quiet Monday equally
with a Black Friday. The one query shape is the cube's; this rule is why.

## 3. AOV divides by purchase events, not by users

`revenue_per_purchaser` = `SUM(revenue_usd) / SUM(purchases)`. Dividing by
`assigned_users` produces revenue per user wearing AOV's name.

## 4. A test window is clipped to the experiment's own start and end

The registry row (`started_on` / `ended_on`) is authoritative. A window running
past `ended_on` produces partial exposure per arm and can manufacture a winner
in a null-effect experiment. Tests run 90 days maximum; a longer implied window
is stated, not silently truncated.

## 5. A lift decision needs dispersion — which does not exist at pooled grain

A pooled rate is one number per arm; it has no variance. Significance comes from
the **day-level** series feeding stats.py (day-level t / CUPED). No answer may
claim "significant" or "winner" from pooled rates alone, and SRM (assignment
share vs intended split) blocks any winner before lift is even read.

## 6. One experiment per question

Lift is only comparable within a single experiment's randomisation. Two
experiment ids in one question is a refusal, not a join.
