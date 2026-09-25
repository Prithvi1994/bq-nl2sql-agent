"""Regression tests: the readout must recover the known ground truth.

Run with:
    .venv/bin/python -m nl2sql.abtest.verify

These are the tests that matter for a demo: not "does the code run" but "does the
pipeline recover the effect that was actually generated, and refuse to ship when the
data is not trustworthy."
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .readout import run_readout

HERE = Path(__file__).resolve().parent
DB = HERE.parent.parent / "abtest_data" / "abtest.duckdb"
TRUTH = HERE.parent.parent / "abtest_data" / "ground_truth.json"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def check(name: str, ok: bool, detail: str = "") -> Tuple[str, bool, str]:
    return (name, ok, detail)


def verify(experiment_id: str) -> List[Tuple[str, bool, str]]:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["experiments"][experiment_id]
    expected_shares = truth["intended_shares"]
    readout = run_readout(
        str(DB),
        experiment_id,
        expected_shares=expected_shares,
        primary_metric=truth["primary_metric"],
        ship_threshold=truth["ship_threshold"],
        guardrail_metric=truth["guardrail_metric"],
        guardrail_tolerance=truth["guardrail_tolerance"],
    )
    a = readout.analysis
    d = readout.decision
    results: List[Tuple[str, bool, str]] = []

    # 1. Denominator must be every assigned user, not just the active ones.
    users = a.get("users", 0)
    results.append(check(
        "assigned-user denominator",
        users == truth["n_users"],
        f"analysed {users:,} of {truth['n_users']:,} assigned",
    ))

    # 2. Each treatment's observed lift must sit near its true lift. The tolerance is
    #    wide because these are random draws; the point is that the sign and rough
    #    magnitude match, and a zero-effect arm stays near zero.
    for variant, metrics in sorted(a.get("lifts", {}).items()):
        true = truth["true_lift_vs_control"][variant]["purchase_rate"]
        obs = metrics["purchase_rate"]["relative_lift"]
        if abs(true) < 1e-9:
            ok = abs(obs) < 0.06
            detail = f"true 0.00% vs observed {_fmt_pct(obs)} (must be within 6pp)"
        else:
            ok = math.copysign(1, obs) == math.copysign(1, true) and abs(obs - true) < 0.08
            detail = f"true {_fmt_pct(true)} vs observed {_fmt_pct(obs)}"
        results.append(check(f"lift recovered: {variant}", ok, detail))

    # 3. SRM must fire exactly when the assignment ratio is wrong.
    srm = a.get("srm", {})
    expect_srm = abs(sum(abs(v - expected_shares.get(k, 0)) for k, v in srm.get("observed", {}).items())) > 0.02
    results.append(check(
        "SRM detection matches ground truth",
        bool(srm.get("detected")) == expect_srm,
        f"detected={srm.get('detected')} expected={expect_srm} observed={srm.get('observed')}",
    ))

    # 4. A detected SRM must force HOLD regardless of effect size.
    if srm.get("detected"):
        results.append(check(
            "SRM forces HOLD",
            d.get("decision") == "HOLD",
            f"decision={d.get('decision')}",
        ))

    # 5. Guardrail regression must be caught on revenue, not just conversion.
    for variant, metrics in sorted(a.get("lifts", {}).items()):
        true_rev = truth["true_lift_vs_control"][variant]["revenue_per_user"]
        obs_rev = metrics["revenue_per_user"]["relative_lift"]
        tol = truth["guardrail_tolerance"]
        broke = obs_rev < tol
        should_broke = true_rev < tol
        if abs(true_rev) > 0.05 or should_broke:
            results.append(check(
                f"guardrail verdict: {variant}",
                broke == should_broke,
                f"true {true_rev:+.2%} observed {obs_rev:+.2%} tolerance {tol:+.2%}",
            ))

    # 6. Every p-value must be a real probability.
    bad_p = [
        (v, m, r["p_value"])
        for v, metrics in a.get("lifts", {}).items()
        for m, r in metrics.items()
        if r["p_value"] is not None and not (0.0 <= r["p_value"] <= 1.0)
    ]
    results.append(check("all p-values in [0,1]", not bad_p, f"bad: {bad_p[:3]}"))

    # 7. CUPED intervals must be ordered; an inverted CI means broken variance maths.
    inverted = [
        (v, m)
        for v, metrics in a.get("lifts", {}).items()
        for m, r in metrics.items()
        if r.get("cuped_ci_low") is not None and r["cuped_ci_low"] > r["cuped_ci_high"]
    ]
    results.append(check("CUPED intervals ordered", not inverted, f"inverted: {inverted[:3]}"))

    # 8. Every query in the report must have executed.
    failed = [q for q in readout.queries if not q["ok"]]
    results.append(check("all report queries executed", not failed, f"failed: {[q['label'] for q in failed]}"))

    # 9. The report must actually use BigQuery surface syntax.
    sql_text = " ".join(q["sql"] for q in readout.queries)
    has_bq = "`" in sql_text and "DECLARE" in sql_text
    results.append(check("report SQL is BigQuery dialect", has_bq, "backticks + DECLARE present"))

    return results


def main() -> int:
    experiments = ["checkout_flow_v2", "pricing_page_copy_v3", "onboarding_email_v1"]
    if not DB.exists():
        print(f"missing {DB}; run: .venv/bin/python -m nl2sql.abtest.build_dataset")
        return 2

    total = passed = 0
    for exp in experiments:
        print(f"\n=== {exp}")
        try:
            results = verify(exp)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            total += 1
            continue
        for name, ok, detail in results:
            total += 1
            passed += int(ok)
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
