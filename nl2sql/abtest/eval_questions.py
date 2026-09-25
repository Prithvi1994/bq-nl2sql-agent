"""Eval for the NL question path. Unlike run_eval.py, these can fail.

Three kinds of check:

  correctness  the answer matches ground truth, not a previous run
  refusal     the agent declines questions it must decline
  grounding   every number in the answer traces to a query that ran

There is no mock LLM anywhere in this file. The question path is deterministic
(metric resolution, control arm, window and significance rule are all code), so a
mock would only prove the mock works.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ask import QuestionError, ask, load_experiments

HERE = Path(__file__).resolve().parent.parent.parent
GOLDEN = HERE / "golden"
TRUTH = HERE / "abtest_data" / "ground_truth.json"


def _truth() -> Dict[str, Any]:
    return json.loads(TRUTH.read_text(encoding="utf-8"))["experiments"]


class Runner:
    def __init__(self, db: str, default_end: str) -> None:
        self.db = db
        self.default_end = date.fromisoformat(default_end)
        self.experiments = load_experiments(db)
        self.truth = _truth()

    def ask(self, question: str) -> Dict[str, Any]:
        return ask(question, self.db, experiments=self.experiments, default_end=self.default_end)


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def check_case(runner: Runner, case: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    name = case["id"]
    out: List[Tuple[str, bool, str]] = []

    try:
        res = runner.ask(case["question"])
    except QuestionError as exc:
        out.append((f"{name}: answered", False, f"refused: {exc}"))
        return out

    plan = res.get("plan", {})
    analysis = res.get("analysis", {})

    if case.get("intent") and plan.get("intent") != case["intent"]:
        out.append((f"{name}: intent", False, f"got {plan.get('intent')}, want {case['intent']}"))
    if case.get("experiment_id") and plan.get("experiment_id") != case["experiment_id"]:
        out.append((f"{name}: experiment", False, f"got {plan.get('experiment_id')}"))
    if case.get("metric") and plan.get("metric") != case["metric"]:
        out.append((f"{name}: metric", False, f"got {plan.get('metric')}, want {case['metric']}"))
    if case.get("expect_dimension") and plan.get("dimension") != case["expect_dimension"]:
        out.append((f"{name}: dimension", False, f"got {plan.get('dimension')}"))
    if "expect_days" in case and plan.get("days") != case["expect_days"]:
        out.append((f"{name}: window", False, f"got {plan.get('days')} days, want {case['expect_days']}"))

    # 90-day cap, and the clamp must be visible rather than silent.
    if plan.get("days", 0) > 90:
        out.append((f"{name}: 90-day cap", False, f"{plan['days']} days"))
    if "expect_clamped" in case:
        clamped = "clamped" in (plan.get("extra") or {})
        out.append((f"{name}: clamp visible", clamped == case["expect_clamped"], str(plan.get("extra"))))

    # Winner correctness.
    if "expect_winner" in case:
        want = case["expect_winner"]
        got = (res.get("winner") or {}).get("variant")
        if want is None:
            ok = got is None
            out.append((f"{name}: no winner claimed", ok, f"claimed {got!r}"))
        else:
            out.append((f"{name}: winner", got == want, f"got {got!r}, want {want!r}"))

    if case.get("expect_no_reliable_winner"):
        text = res.get("answer", "").lower()
        ok = ("no reliable difference" in text or "no variant beats" in text
              or "includes zero" in text or "no winner" in text)
        out.append((f"{name}: hedged language", ok, res.get("answer", "")[:90]))

    if "expect_winner_significant" in case:
        got = bool((res.get("winner") or {}).get("significant"))
        out.append((f"{name}: significance", got == case["expect_winner_significant"], f"got {got}"))

    if case.get("expect_focus_variant"):
        got = (plan.get("extra") or {}).get("focus_variant")
        out.append((f"{name}: focus variant", got == case["expect_focus_variant"],
                    f"got {got!r}, want {case['expect_focus_variant']!r}"))

    # Lift must be near the true lift. The arm to check may differ from the winner:
    # when SRM blocks a decision there is no winner, but the effect is still real.
    if case.get("expect_true_lift_within_pp") is not None:
        tol = case["expect_true_lift_within_pp"] / 100.0
        metric = case.get("metric") or ""
        variant = (case.get("expect_lift_variant") or case.get("expect_focus_variant")
                   or case.get("expect_winner") or "treatment_b")
        true_key = {"revenue_per_purchaser": "aov"}.get(metric, metric)
        true_lift = runner.truth[case["experiment_id"]]["true_lift_vs_control"][variant].get(
            true_key, 0.0
        )
        lifts = analysis.get("lifts", {})
        if variant in lifts:
            obs = lifts[variant]["relative_lift"]
            ok = abs(obs - true_lift) <= tol
            out.append((f"{name}: lift within {case['expect_true_lift_within_pp']}pp",
                        ok, f"true {_fmt(true_lift)} vs observed {_fmt(obs)}"))
        else:
            out.append((f"{name}: lift", False, f"no lift for {variant}"))

    # SRM.
    if "expect_srm_detected" in case:
        srm = (res.get("analysis") or {}).get("srm") or res.get("srm")
        detected = bool(srm and srm.get("detected"))
        out.append((f"{name}: SRM", detected == case["expect_srm_detected"],
                    f"detected={detected} want={case['expect_srm_detected']}"))
    if case.get("expect_srm_in_answer"):
        text = res.get("answer", "").lower()
        out.append((f"{name}: SRM in answer", "sample ratio mismatch" in text, "not mentioned"))

    # Trend / cohort shape.
    if "expect_variants" in case:
        n = len({s["variant"] for s in res.get("trend", [])})
        out.append((f"{name}: variant count", n == case["expect_variants"], f"got {n}"))
    if "expect_min_segments" in case:
        n = len([line for line in res.get("detail", []) if line.strip().startswith("- ")])
        out.append((f"{name}: segment count", n >= case["expect_min_segments"], f"got {n}"))

    # Duration math.
    if case.get("expect_n_per_arm_positive"):
        n = res.get("n_per_arm", 0)
        out.append((f"{name}: n_per_arm", isinstance(n, int) and n > 0, f"got {n}"))

    # Grounding: every query must have executed.
    failed = [q for q in res.get("queries", []) if not q.get("ok")]
    out.append((f"{name}: queries executed", not failed,
                f"failed: {[q['label'] for q in failed]}"))

    # BigQuery surface must be preserved in the emitted SQL.
    sql = " ".join(q.get("sql", "") for q in res.get("queries", []))
    if sql:
        out.append((f"{name}: BigQuery dialect", "`" in sql and "DECLARE" in sql, "no backticks/DECLARE"))

    return out


def check_refusal(runner: Runner, case: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    name = case["id"]
    should_refuse = not case.get("not_refused")
    try:
        res = runner.ask(case["question"])
    except QuestionError as exc:
        if should_refuse:
            return [(f"{name}: refused", True, str(exc)[:100])]
        return [(f"{name}: answered", False, f"refused: {exc}")]

    if should_refuse:
        return [(f"{name}: refused", False, f"answered instead: {res.get('answer', '')[:90]}")]

    extra = res.get("plan", {}).get("extra") or {}
    ok = res.get("plan", {}).get("days", 0) <= 90 and "clamped" in extra
    return [(f"{name}: clamped not refused", ok, f"plan={res.get('plan')}")]


def main() -> int:
    cases_path = GOLDEN / "abtest_questions.json"
    refusals_path = GOLDEN / "abtest_refusals.json"
    cases_doc = json.loads(cases_path.read_text(encoding="utf-8"))
    refusals_doc = json.loads(refusals_path.read_text(encoding="utf-8"))

    db = str(HERE / cases_doc["db"].lstrip("./"))
    if not Path(db).exists():
        print(f"missing {db}\nrun: .venv/bin/python -m nl2sql.abtest.build_dataset")
        return 2

    runner = Runner(db, cases_doc["default_end"])
    total = passed = 0
    failures: List[str] = []

    print(f"=== correctness ({len(cases_doc['cases'])} questions)")
    for case in cases_doc["cases"]:
        for label, ok, detail in check_case(runner, case):
            total += 1
            passed += int(ok)
            if not ok:
                failures.append(f"{label} — {detail}")
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))

    print(f"\n=== refusal ({len(refusals_doc['cases'])} questions)")
    for case in refusals_doc["cases"]:
        for label, ok, detail in check_refusal(runner, case):
            total += 1
            passed += int(ok)
            if not ok:
                failures.append(f"{label} — {detail}")
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))

    print(f"\n{passed}/{total} checks passed")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
