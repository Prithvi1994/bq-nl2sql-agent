"""Question eval for the Wren-native path.

Scores on executed results and behaviour, not on plan internals. The old checks
read the deterministic QueryPlan (intent / metric / days); those fields do not
exist when the LLM writes the SQL directly. What replaces them: the same ground
truth, the refusal discipline, and a grounding assertion that the planned SQL
came out of Wren's gate.

An LLM eval must be able to fail. A wrong-but-plausible number is a failure here
just as it was on the deterministic path — the oracle is the injected effect
size, unchanged.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .answer import QuestionError, ask

HERE = Path(__file__).resolve().parent.parent.parent
GOLDEN = HERE / "golden"


class Runner:
    """Runs the Wren-native answer path: LLM writes SQL, the gate checks it."""

    def __init__(self, db: str) -> None:
        self.db = db
        self.truth = _truth()

    def ask(self, question: str) -> Dict[str, Any]:
        return ask(question, self.db)


def _truth() -> Dict[str, Any]:
    path = HERE / "abtest_data" / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))["experiments"]


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _guess_metric_key(question: str) -> str:
    L = question.lower()
    if "session" in L:
        return "sessions_per_user"
    if "aov" in L or "order value" in L:
        return "aov"
    if "purchase" in L or "conversion" in L:
        return "purchase_rate"
    if "cart" in L:
        return "add_to_cart_rate"
    if "pageview" in L:
        return "sessions_per_user"
    return "revenue_per_user"


def check_case(runner: Runner, case: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    name = case["id"]
    out: List[Tuple[str, bool, str]] = []

    try:
        res = runner.ask(case["question"])
    except QuestionError as exc:
        out.append((f"{name}: answered", False, f"gate refused: {str(exc)[:110]}"))
        return out
    except Exception as exc:  # noqa: BLE001 - one broken case must not kill the run
        out.append((f"{name}: answer crashed", False, f"{type(exc).__name__}: {str(exc)[:80]}"))
        return out

    if case.get("expect_refusal"):
        ok = bool(res.get("refused"))
        out.append((f"{name}: refusal", ok,
                    f"answered instead: {res.get('answer', '')[:80]}" if not ok
                    else res.get("answer", "")[:70]))
        return out

    # Grounding: the query that ran was planned by Wren and passed the gate.
    q = (res.get("queries") or [{}])[-1]
    out.append((
        f"{name}: grounded (Wren plan + gate)",
        bool(res.get("planned_sql")) and q.get("invariants") == "passed",
        "planned SQL missing or the gate did not record a pass",
    ))

    # Lift recovered from the query's own per-user rows, against the injected truth.
    lifts = (res.get("analysis") or {}).get("lifts") or {}
    if case.get("expect_true_lift_within_pp") is not None:
        tol = case["expect_true_lift_within_pp"] / 100.0
        variant = (case.get("expect_lift_variant") or case.get("expect_winner")
                   or "treatment_b")
        key = case.get("metric") or _guess_metric_key(case["question"])
        true_key = {"revenue_per_purchaser": "aov"}.get(key, key)
        true_lift = runner.truth[case["experiment_id"]]["true_lift_vs_control"][variant].get(
            true_key, 0.0
        )
        if variant in lifts:
            obs = lifts[variant]["relative_lift"]
            out.append((
                f"{name}: lift within {case['expect_true_lift_within_pp']}pp",
                math.isfinite(obs) and abs(obs - true_lift) <= tol,
                f"true {_fmt(true_lift)} vs observed {_fmt(obs)}",
            ))
        else:
            out.append((f"{name}: lift", False,
                        f"no lift computed for {variant} — the model returned "
                        "grouped rows instead of per-user rows"))

    if "expect_winner" in case:
        got = (res.get("winner") or {}).get("variant")
        want = case["expect_winner"]
        if want is None:
            out.append((f"{name}: no winner claimed", got is None, f"claimed {got!r}"))
        else:
            out.append((f"{name}: winner", got == want, f"got {got!r}, want {want!r}"))

    if "expect_srm_detected" in case:
        got = bool((res.get("srm") or {}).get("detected"))
        out.append((f"{name}: srm detected", got == case["expect_srm_detected"],
                    f"got {got}, want {case['expect_srm_detected']}"))

    return out


def check_refusal(runner: Runner, case: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    name = case["id"]
    expect_refusal = case.get("expect_refusal", True)
    try:
        res = runner.ask(case["question"])
    except QuestionError as exc:
        return [(f"{name}: refused", expect_refusal, str(exc)[:100])]
    except Exception as exc:  # noqa: BLE001
        return [(f"{name}: crashed", False, f"{type(exc).__name__}: {str(exc)[:80]}")]
    refused = bool(res.get("refused"))
    return [(f"{name}: refused", refused == expect_refusal,
             res.get("answer", "")[:90])]


def main() -> int:
    cases_path = GOLDEN / "abtest_questions.json"
    refusals_path = GOLDEN / "abtest_refusals.json"
    cases_doc = json.loads(cases_path.read_text(encoding="utf-8"))
    refusals_doc = json.loads(refusals_path.read_text(encoding="utf-8"))

    db = str(HERE / cases_doc["db"].lstrip("./"))
    if not Path(db).exists():
        print(f"missing {db}\nrun: .venv/bin/python -m nl2sql.abtest.build_dataset")
        return 2

    runner = Runner(db)
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
    sys_exit = main()
    raise SystemExit(sys_exit)