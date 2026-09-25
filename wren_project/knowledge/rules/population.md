# Population rules

These are policies, not schema facts. The MDL encodes what each measure *is*; this
file encodes which measure is legal in which context. An agent that follows the MDL
but not this file will produce a valid, plausible, wrong number.

## 1. The denominator is assigned users, never active users

`experiment_daily` (one row per user per *active* day) is not the population.
A user who was randomised and never came back has no row in it.

All per-user rates divide by `assigned_users`, which is `COUNT(*)` over
`experiment_user` — one row per assignment, zero-filled.

**Symptom of getting this wrong:** every rate is inflated by roughly the
active-user share. In our 30,000-user test with a 50% active rate, a purchase
rate computed over active users reads roughly double the truth, and the lift
still looks significant. No syntax check or result-shape check catches this.

**Never** do this:
```sql
SELECT SAFE_DIVIDE(COUNT(DISTINCT IF(purchases>0,user_id,NULL)), COUNT(DISTINCT user_id))
FROM experiment_daily          -- denominator excludes inactive users
```

## 2. AOV divides by purchase events, not by users and not by assigned users

`revenue_per_purchaser` uses `SUM(purchases)` in the denominator. Dividing by
`assigned_users` produces revenue per user wearing AOV's name — a different
metric that will differ by the non-purchasing share, usually a large amount.

**Never** do this:
```sql
SAFE_DIVIDE(SUM(revenue_usd), assigned_users)   -- this is revenue_per_user
```

## 3. "Purchasing users" is a distinct-user count, not a sum of events

A user who buys three times is one purchasing user and three purchase events.
`COUNT(DISTINCT IF(purchases > 0, user_id, NULL))` is the conversion denominator;
`SUM(purchases)` is the event count. They differ by repeat rate.

## 4. A test window is clipped to the experiment's own start and end

`dim_experiments.started_on` / `ended_on` are authoritative. A requested window
that runs past `ended_on` produces partial exposure per arm, which manufactures a
winner in a null-effect experiment.

Tests run 90 days maximum. If a question implies a longer window, say so rather
than silently returning the truncated result.

## 5. `days_active` is not a rate

`days_active` is a count of active days summed over users. To get a per-user
active-day rate, divide by `assigned_users`. To get an active-day rate, divide by
the window length. Dividing by nothing gives a meaningless large number.
