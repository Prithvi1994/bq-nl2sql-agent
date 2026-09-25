"""LLM BigQuery SQL generation eval.

Scores whether *executed results* are correct, not whether the SQL string resembles a
reference. String matching is the wrong instrument here: two queries can differ in
every character and return the same rows, and a query can look right and return a
denominator that silently drops inactive users.

The truth for each case is computed independently in Python from the same DuckDB
mirror, using SQL written by hand. The model never sees it.

Run:
    set -a && . /opt/data/.env && set +a
    .venv/bin/python -m nl2sql.abtest.eval_sqlgen
    .venv/bin/python -m nl2sql.abtest.eval_sqlgen --model space-bunny-free --n 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
CASES_PATH = HERE / "golden" / "sqlgen_cases.json"

SYSTEM = """\
You write BigQuery SQL against a governed semantic layer.

## Models you can query (use these names, never the underlying tables)
- `experiment_daily` — one row per user per day of activity, with `variant`, `country`,
  `platform` already joined in. Columns: metric_date DATE, user_id STRING,
  experiment_id STRING, variant STRING, country STRING, platform STRING,
  phase STRING, sessions INT64, pageviews INT64, add_to_cart INT64,
  purchases INT64, revenue_usd FLOAT64, session_duration_s FLOAT64
- `assigned_population` — one row per user per experiment. Use this for any
  denominator.

## Rules
- `phase` is 'pre' (before the test) or 'test' (the test window). Experiment
  questions mean phase = 'test' unless stated otherwise.
- Count DISTINCT users with COUNT(DISTINCT user_id). Never COUNT(*) for a user count.
- "users who purchased" means COUNT(DISTINCT IF(purchases > 0, user_id, NULL)), which
  is not the same as SUM(purchases).
- Active users are not the same as assigned users. Do not substitute one for the other.
- Reply with the SQL only: no prose, no explanation, no markdown fences.
"""

MAX_REPAIRS = 1

TRUTH_SQL: Dict[str, str] = {
    "variant_totals": """
        SELECT variant, SUM(revenue_usd) AS v
        FROM fact_daily_assigned
        WHERE experiment_id = 'checkout_flow_v2' AND phase = 'test'
        GROUP BY variant ORDER BY variant
    """,
    "country_totals": """
        SELECT country, SUM(revenue_usd) AS v
        FROM fact_daily_assigned
        WHERE experiment_id = 'checkout_flow_v2' AND phase = 'test'
        GROUP BY country ORDER BY country
    """,
    "purchasing_users_by_variant": """
        SELECT variant, COUNT(DISTINCT IF(purchases > 0, user_id, NULL)) AS v
        FROM fact_daily_assigned
        WHERE experiment_id = 'checkout_flow_v2' AND phase = 'test'
        GROUP BY variant ORDER BY variant
    """,
    "active_users_by_variant": """
        SELECT variant, COUNT(DISTINCT user_id) AS v
        FROM fact_daily_assigned
        WHERE experiment_id = 'checkout_flow_v2' AND phase = 'test'
        GROUP BY variant ORDER BY variant
    """,
    "first_7_days_total": """
        SELECT SUM(revenue_usd) AS v FROM fact_daily_assigned
        WHERE experiment_id = 'checkout_flow_v2' AND phase = 'test'
          AND metric_date < DATE '2026-01-12'
    """,
    "variant_totals_onboarding": """
        SELECT variant, SUM(revenue_usd) AS v
        FROM fact_daily_assigned
        WHERE experiment_id = 'onboarding_email_v1' AND phase = 'test'
        GROUP BY variant ORDER BY variant
    """,
}


def _norm_rows(rows: Sequence[Sequence[Any]], decimals: int = 2) -> list[tuple]:
    out = []
    for r in rows:
        out.append(
            tuple(
                (round(float(v), decimals) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for v in r
            )
        )
    return sorted(out, key=lambda t: tuple(str(x) for x in t))


def extract_sql(text: str) -> str:
    """Pull SQL out of a model response. Tolerates fences and prose."""
    if not text:
        return ""
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    t = text.strip()
    if t.upper().startswith(("SELECT", "WITH")):
        return t
    m = re.search(r"((?:SELECT|WITH)\s.+$)", t, re.S | re.I)
    return m.group(1).strip() if m else t


class Scorer:
    def __init__(self, db: str) -> None:
        self.db = db
        self._truth_cache: Dict[str, list[tuple]] = {}

    def truth(self, check: str) -> list[tuple]:
        if check not in self._truth_cache:
            from ..bq_exec import run_bq_query
            res = run_bq_query(TRUTH_SQL[check], self.db, row_limit=10_000)
            if not res.ok:
                raise RuntimeError(f"truth query {check} failed: {res.error}")
            self._truth_cache[check] = _norm_rows(res.rows)
        return self._truth_cache[check]

    def score(self, case: Dict[str, Any], sql: str) -> Dict[str, Any]:
        from ..bq_exec import run_bq_query

        findings: List[str] = []
        # Model names must be mapped to the mirror's physical views before execution.
        physical = sql.replace("experiment_daily", "fact_daily_assigned") \
                       .replace("assigned_population", "fact_user_assignments")
        res = run_bq_query(physical, self.db, row_limit=10_000)
        if not res.ok:
            return {"ok": False, "kind": "exec_error", "detail": res.error[:200],
                    "sql": sql, "findings": findings}

        got = _norm_rows(res.rows)
        want = self.truth(case["check"])

        if not sql:
            findings.append("no SQL extracted")
        for ref in case.get("must_reference", []):
            if ref not in sql:
                findings.append(f"missing reference to {ref}")
        for ref in case.get("forbid_reference", []):
            if ref in sql:
                findings.append(f"forbidden reference to {ref}")

        if got == want:
            findings.append("result matches truth")
            return {"ok": True, "kind": "exact", "detail": "", "sql": sql,
                    "findings": findings, "rows": got}

        # A numeric mismatch is the dangerous case: valid SQL, plausible output, wrong.
        if len(got) == len(want):
            close = True
            diffs = []
            for g, w in zip(got, want):
                for gv, wv in zip(g, w):
                    if isinstance(gv, (int, float)) and isinstance(wv, (int, float)):
                        if abs(float(gv) - float(wv)) > max(abs(float(wv)) * 1e-6, 0.01):
                            close = False
                            diffs.append(f"{wv} != {gv}")
                    elif gv != wv:
                        close = False
                        diffs.append(f"{wv!r} != {gv!r}")
            if close:
                findings.append("result matches truth within tolerance")
                return {"ok": True, "kind": "tolerance", "detail": "", "sql": sql,
                        "findings": findings, "rows": got}
            return {"ok": False, "kind": "wrong_values", "detail": "; ".join(diffs[:4]),
                    "sql": sql, "findings": findings, "rows": got}

        return {"ok": False, "kind": "wrong_shape",
                "detail": f"got {len(got)} rows, want {len(want)}",
                "sql": sql, "findings": findings, "rows": got}


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval LLM BigQuery SQL generation.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--db", default=str(HERE / "abtest_data" / "abtest.duckdb"))
    ap.add_argument("--n", type=int, default=1, help="repeats per case, for variance")
    ap.add_argument("--verbose", "-v", action="store_true", help="print generated SQL")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"missing {args.db}\nrun: .venv/bin/python -m nl2sql.abtest.build_dataset")
        return 2

    from .chat_llm import LLMError, create_llm

    try:
        llm = create_llm(args.model, base_url=args.base_url)
    except LLMError as exc:
        print(f"Cannot start: {exc}")
        return 2

    doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    scorer = Scorer(args.db)

    print(f"model: {llm.name}   cases: {len(doc['cases'])}   repeats: {args.n}\n")
    total = passed = 0
    per_case: Dict[str, List[bool]] = {}
    latencies: List[int] = []

    for case in doc["cases"]:
        results = []
        for attempt in range(args.n):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": case["question"]},
            ]
            try:
                raw = llm.complete(messages)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {case['id']}: {type(exc).__name__}: {exc}")
                results.append({"ok": False, "kind": "model_error", "detail": str(exc)[:200],
                                "sql": "", "findings": []})
                continue
            lat = llm.last_usage.get("elapsed_ms") or 0
            if lat:
                latencies.append(lat)
            sql = extract_sql(raw)
            outcome = scorer.score(case, sql)

            # A repair round, because that is the real path: a bound variable it
            # could not resolve, a dialect slip. One attempt to fix, then give up.
            if not outcome["ok"] and outcome["kind"] == "exec_error" and MAX_REPAIRS:
                repair_msgs = messages + [
                    {"role": "assistant", "content": sql},
                    {"role": "user", "content": (
                        f"That query failed: {outcome['detail']}\n"
                        "Fix it. Remember: this is BigQuery, and you cannot reference "
                        "columns or variables that do not exist. Reply with the corrected "
                        "SQL only."
                    )},
                ]
                try:
                    fixed = extract_sql(llm.complete(repair_msgs))
                    repaired = scorer.score(case, fixed)
                    repaired["repaired"] = True
                    repaired["failed_sql"] = sql
                    outcome = repaired
                except Exception:  # noqa: BLE001 - keep the original failure
                    pass

            results.append(outcome)
            if args.verbose:
                print(f"\n  --- {case['id']} attempt {attempt + 1} SQL\n{sql}\n")
        oks = [r["ok"] for r in results]
        per_case[case["id"]] = oks
        for ok in oks:
            total += 1
            passed += int(ok)
        best = results[0]
        mark = "PASS" if all(oks) else ("FLAKY" if any(oks) else "FAIL")
        print(f"  [{mark}] {case['id']}  ({sum(oks)}/{len(oks)})  kind={best['kind']}")
        if not all(oks) or args.verbose:
            for line in best["findings"]:
                print(f"         - {line}")
            if best["detail"]:
                print(f"         ! {best['detail']}")
            if best["sql"]:
                print(f"         sql: {' '.join(best['sql'].split())[:180]}")

    # Count successes, not entries: summing every element would report 6/6 even when
    # one case failed, because `sum(1 for ok in oks)` counts the False too.
    exact = sum(1 for oks in per_case.values() for ok in oks if ok)
    print(f"\n{'=' * 60}")
    print(f"result-match accuracy : {exact}/{total} = {100.0 * exact / max(total, 1):.1f}%")
    stable = sum(1 for oks in per_case.values() if all(oks) or not any(oks))
    print(f"stable across repeats : {stable}/{len(per_case)} cases")
    if latencies:
        print(f"median latency        : {int(np.median(latencies))} ms")
    print(f"model                 : {llm.name}")
    print("=" * 60)
    return 0 if exact == total else 1


if __name__ == "__main__":
    sys.exit(main())
