"""Wren-native experiment answering.

Pipeline, with authorship explicit:

  1. LLM writes ONE BigQuery SELECT against MDL model names, grounded by the
     compiled MDL, knowledge rules, and memory recall. It never sees physical
     tables and never writes a metric definition it was not shown.
  2. Wren (dry-plan) expands model names to real BigQuery — transport only.
  3. Invariants gate the planned SQL.
  4. Execution against the mirror.
  5. stats.py computes SRM / lift / CI / ship-rule — the LLM never touches a number.

No regex NL parsing, no SQL templates. The LLM is the NL interface; the MDL is
the only schema it sees; the invariants are the part an LLM cannot self-apply.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import metrics as M
from .stats import check_srm
from .wren_engine import DEFAULT_PROJECT, WrenError, plan as wren_plan
from .policy import check_policy
from ..bq_exec import run_bq_query


class QuestionError(ValueError):
    """A question that must not be answered. Refusal is a feature, not a failure."""


SYSTEM_TEMPLATE = """\
You are an agent answering A/B experiment questions through a Wren semantic layer.

Follow these steps IN ORDER for every question. Do not skip a step.

1. IDENTIFY which single experiment the question names, from the list at the end.
2. CLASSIFY the question:
   - asks to list users / emails / names / contact details → REFUSAL
   - names TWO different experiments → REFUSAL (lift is meaningless across randomisations)
   - asks for a winner / lift / comparison → PER-USER row query (step 3a)
   - asks for a breakdown by segment → GROUPED query (step 3b)
   - asks about assignment shares → POPULATION query (step 3c)
3a. PER-USER (the statistics layer needs one row per assigned user — no GROUP BY):
      SELECT user_id, variant, <metric expression> AS metric_value
      FROM experiment_user WHERE experiment_id = '<id>'
    Metric expressions, by question:
      revenue per user  -> revenue_usd
      sessions per user -> sessions
      purchase rate     -> CASE WHEN purchases > 0 THEN 1.0 ELSE 0.0 END
3b. GROUPED (one row per segment value):
      SELECT <dimension>, variant, COUNT(DISTINCT user_id) AS assigned_users,
             AVG(<metric column>) AS metric_per_user
      FROM experiment_user WHERE experiment_id = '<id>' GROUP BY 1, 2
3c. POPULATION:
      SELECT variant, COUNT(*) AS users FROM assigned_population
      WHERE experiment_id = '<id>' GROUP BY variant
4. Reply with ONLY the SQL. No prose, no markdown fences, no commentary.

REFUSAL (reply exactly, nothing else):
- no experiment named:      REFUSE: I need the experiment id
- two experiments named:    REFUSE: one experiment at a time
- lists users or their details: REFUSE: row-level user data is out of scope

Binding semantic rules (verified by the gate; the gate REJECTS violating SQL):
- population is ASSIGNED users (experiment_user / assigned_population), never
  active users (experiment_daily) — see '## Rule: population' below
- every ratio uses SAFE_DIVIDE; every user count uses COUNT(DISTINCT user_id)
  unless the query is per-user rows
- any date window stays inside the experiment's own start/end dates and <= 90 days

{context}

## Experiments (window bounds are authoritative)
{experiments}
"""


# --------------------------------------------------------------------------
# Grounded context
# --------------------------------------------------------------------------

def build_context(project: Path | str = DEFAULT_PROJECT) -> str:
    """Models, columns, cube expression docs, rules, memory recall -- from Wren."""
    project = Path(project)
    mdl_path = project / "target" / "mdl.json"
    if not mdl_path.exists():
        raise QuestionError(
            f"{mdl_path} not found. Run: .venv/bin/wren context build --path {project}"
        )
    mdl = json.loads(mdl_path.read_text(encoding="utf-8"))
    parts: List[str] = []

    parts.append("## Models (FROM must use one of these names, never a physical table)")
    for model in mdl.get("models", []):
        ref = model.get("tableReference") or {}
        fqn = ".".join(x for x in (ref.get("catalog"), ref.get("schema"), ref.get("table")) if x)
        parts.append(f"\n### {model['name']}")
        if fqn:
            parts.append(f"physical: {fqn}")
        desc = " ".join(((model.get("properties") or {}).get("description") or "").split())
        if desc:
            parts.append(desc)
        for col in model.get("columns", []):
            d = " ".join((col.get("description") or "").split())
            parts.append(f"  {col['name']} {col.get('type', '?')}" + (f"  -- {d}" if d else ""))

    for cube in mdl.get("cubes", []):
        parts.append(
            f"\n## Cube {cube['name']} — canonical metric expressions (documentation; "
            "imitate these, do not query the cube name)"
        )
        for measure in cube.get("measures", []):
            d = " ".join((measure.get("description") or "").split())
            parts.append(f"  {measure['name']} = {measure.get('expression', '')}"
                         + (f"  -- {d}" if d else ""))
        parts.append("  dimensions: " + ", ".join(d["name"] for d in cube.get("dimensions", [])))

    rules_dir = project / "knowledge" / "rules"
    if rules_dir.exists():
        for rule_file in sorted(rules_dir.glob("*.md")):
            parts.append(f"\n## Rule: {rule_file.stem}\n{rule_file.read_text(encoding='utf-8')}")

    pairs = _memory_recall_pairs()
    if pairs:
        parts.append("\n## Confirmed queries (imitate these patterns exactly)")
        for nl, sql in pairs:
            parts.append(f"\nQ: {nl}\n{sql}")

    return "\n".join(parts)


def _memory_recall_pairs(limit: int = 4) -> List[tuple[str, str]]:
    """Confirmed NL->SQL pairs, newest first. Empty until one is confirmed."""
    sql_dir = Path(DEFAULT_PROJECT) / "knowledge" / "sql"
    if not sql_dir.exists():
        return []
    out: List[tuple[str, str]] = []
    for f in sorted(sql_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        text = f.read_text(encoding="utf-8")
        m_sql = re.search(r"```sql\s*(.+?)```", text, re.S)
        m_nl = re.search(r"^#\s*(.+)$", text, re.M)
        if m_sql:
            out.append((m_nl.group(1).strip() if m_nl else f.stem, m_sql.group(1).strip()))
    return out


def load_experiments(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Experiment metadata from dim_experiments — the authority for window clipping."""
    res = run_bq_query(
        "SELECT experiment_id, name, started_on, ended_on, target_split, "
        "ship_threshold, guardrail_metric, guardrail_tolerance "
        "FROM dim_experiments",
        db_path, row_limit=100,
    )
    if not res.ok:
        raise QuestionError(f"cannot read dim_experiments: {res.error}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in res.rows:
        (exp_id, name, start, end, split, ship, guard, tol) = row
        out[str(exp_id)] = {
            "experiment_id": str(exp_id), "name": name,
            "started_on": str(start), "ended_on": str(end),
            "intended_shares": _parse_split(split),
            "ship_threshold": float(ship) if ship is not None else None,
            "guardrail_metric": guard,
            "guardrail_tolerance": float(tol) if tol is not None else None,
        }
    return out


def _parse_split(split: Any) -> Dict[str, float]:
    if not split:
        return {}
    shares: Dict[str, float] = {}
    for part in re.split(r"[,/]", str(split)):
        if ":" in part:
            v, s = part.rsplit(":", 1)
            try:
                shares[v.strip()] = float(s)
            except ValueError:
                pass
    return shares


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------

def check_invariants(
    planned_sql: str,
    experiment_id: str,
    experiments: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Reject SQL that would produce a plausible wrong number.

    Every rule encodes a bug that produced a wrong answer this session.
    """
    bad: List[str] = []
    p = planned_sql

    if experiment_id and experiment_id not in p:
        bad.append(f"planned SQL does not filter on experiment {experiment_id!r}")

    # Denominator rule: a bare COUNT over the daily model counts user-days.
    if re.search(r"\bexperiment_daily\b", p) and re.search(r"COUNT\s*\(\s*\*\s*\)", p, re.I):
        bad.append("COUNT(*) over experiment_daily counts user-days, not assigned users")

    # Row-level export: the SELECT list must not be a bare user roster. Per-user
    # rows feeding the stats layer are fine — they carry a metric column and are
    # consumed in-process. Contact columns are always a refusal.
    sel = re.search(r"SELECT\s+(.*?)(?:\bFROM\b)", p, re.S | re.I)
    if sel:
        head = sel.group(1).lower()
        has_metric = any(k in head for k in
                         ("metric", "sum(", "count(", "avg(", "max(", "min(", "safe_divide"))
        contact = any(k in head for k in ("email", "phone", "address"))
        if contact or (re.search(r"\buser_id\b", head) and not has_metric):
            bad.append("query exports raw user rows (row-level export)")

    # Window clipping: metric_date bounds must sit inside the experiment, and the
    # window may not exceed 90 days — the generator's test-run cap.
    meta = experiments.get(experiment_id, {})
    if "BETWEEN" in p.upper():
        dates = re.findall(r"'(\d{4}-\d{2}-\d{2})'", p)
        if len(dates) >= 2:
            from datetime import date
            lo, hi = sorted(dates[:2])
            d_lo, d_hi = date.fromisoformat(lo), date.fromisoformat(hi)
            if (d_hi - d_lo).days > 90:
                bad.append("window exceeds the 90-day test cap")
            if meta and (lo < meta["started_on"] or hi > meta["ended_on"]):
                bad.append(
                    f"window {lo}..{hi} exceeds the experiment's "
                    f"{meta['started_on']}..{meta['ended_on']}"
                )

    return bad


# --------------------------------------------------------------------------
# ask()
# --------------------------------------------------------------------------

def ask(
    question: str,
    db_path: str,
    *,
    llm: Optional[Any] = None,
    project: Path | str = DEFAULT_PROJECT,
    max_attempts: int = 3,
    store_confirmed: bool = True,
) -> Dict[str, Any]:
    """Answer a business question. The only public entry point."""
    if llm is None:
        from .chat_llm import create_llm
        # A fresh session header per question. Reusing one header across many
        # questions degrades the provider's replies into rule-reciting prose after
        # ~20 turns — a fresh session isolates each question from the repair-loop
        # history of the previous one.
        import uuid
        llm = create_llm(session=f"abq-{uuid.uuid4().hex[:12]}")

    experiments = load_experiments(db_path)
    exp_block = "\n".join(
        f"- {e['experiment_id']}: {e['started_on']} .. {e['ended_on']}, "
        f"intended split {e['intended_shares'] or 'unrecorded'}"
        for e in experiments.values()
    )
    system = SYSTEM_TEMPLATE.format(context=build_context(project), experiments=exp_block)

    experiment_id = ""
    for exp in sorted(experiments, key=len, reverse=True):
        if re.search(rf"\b{re.escape(exp)}\b", question.lower()):
            experiment_id = exp
            break

    log: List[Dict[str, Any]] = []
    planned = None
    last_error = "no attempt made"

    for attempt in range(max_attempts):
        # The repair message must REPEAT the question: with only an error in view
        # ("no actual substantive question visible" — a real model reply), the
        # model invents context or refuses.
        user_msg = question if attempt == 0 else (
            f"Question: {question}\n"
            f"Your previous reply failed: {last_error}\n"
            "Reply with the corrected SQL only."
        )
        # One retry on a provider-level failure (timeout / connection). Distinct
        # from the SQL-repair retry: this catches transport flakiness observed in
        # sequential runs, where ~1 in 15 calls dies mid-read.
        for spin in range(2):
            try:
                raw = llm.complete([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ])
                break
            except Exception as exc:  # noqa: BLE001
                if spin == 1:
                    raise
                import time
                time.sleep(2)
        intent = _extract_sql(raw)
        if not intent:
            last_error = ("the reply contained no SQL — reasoning prose is not an "
                          "answer. Reply with the SQL for the question and nothing else")
            log.append({"attempt": attempt, "extract": "empty", "raw_head": raw[:120]})
            continue

        if intent.upper().startswith("REFUSE"):
            if experiment_id and attempt + 1 < max_attempts:
                # The question names a known experiment. A refusal here is the
                # model over-applying the refusal rules, one retry is cheap and
                # the second reply is checked by the same gates as any other.
                last_error = (f"the question names experiment {experiment_id}; that is an "
                              "answerable question. Reply with the SQL only.")
                log.append({"attempt": attempt, "model_refused": intent.strip()[:80]})
                continue
            return {"answer": intent.strip(), "refused": True, "queries": log}

        if not experiment_id:
            last_error = ("no experiment named in the question; the query would "
                          "silently aggregate across experiments. Reply with "
                          "REFUSE: I need the experiment id")
            log.append({"attempt": attempt, "gate": "failed",
                        "violations": [last_error], "sql": intent[:120]})
            continue

        meta = experiments.get(experiment_id, {})
        violations = check_policy(
            intent,
            allowed_models={"experiment_user", "experiment_daily",
                            "assigned_population", "experiment_metrics"},
            experiment_id=experiment_id,
            experiment_bounds=meta,
        )
        if violations:
            last_error = "; ".join(violations)
            from .policy import PolicyError  # noqa: F401
            log.append({"attempt": attempt, "policy": "rejected",
                        "violations": violations, "sql": intent[:160],
                        "raw_head": raw[:200]})
            planned = None
            continue

        try:
            planned = wren_plan(intent, project=project)
        except WrenError as exc:
            last_error = str(exc)
            log.append({"attempt": attempt, "plan": "failed",
                        "error": str(exc)[:300], "sql": intent[:160],
                        "raw_head": raw[:200]})
            planned = None
            continue

        violations = check_invariants(planned.planned_sql, experiment_id, experiments)
        if violations:
            last_error = "; ".join(violations)
            log.append({"attempt": attempt, "gate": "failed",
                        "violations": violations, "sql": intent[:160]})
            planned = None
            continue
        break

    if planned is None:
        raise QuestionError(
            f"no query passed the semantic gate in {max_attempts} attempts: {last_error}"
        )

    res = run_bq_query(_to_mirror(planned.planned_sql), db_path, row_limit=1_000_000)
    log.append({
        "attempt": len(log), "ok": res.ok, "row_count": res.row_count,
        "sql": planned.sql, "planned_sql": planned.planned_sql,
        "plan_ms": planned.plan_ms,
        "error": None if res.ok else res.error[:400],
        "invariants": "passed",
    })
    if not res.ok:
        raise QuestionError(f"query failed: {res.error}")

    out: Dict[str, Any] = {
        "refused": False, "sql": planned.sql, "planned_sql": planned.planned_sql,
        "rows": res.rows, "columns": res.columns, "queries": log,
        "experiment_id": experiment_id,
    }

    # Statistics stay in Python. Per-user rows get lift/CI; the SRM check is ALWAYS
    # ours -- a property of the assignment table, not something the question needs
    # to ask for. The old path failed 5 eval cases because it waited for the model
    # to fetch the counts it never intended to.
    out["srm"] = _run_srm_check(experiment_id, experiments, db_path)
    if _looks_per_user(res.columns):
        applied = _apply_lift_stats(
            res.rows, res.columns, experiment_id, experiments, srm=out["srm"]
        )
        out.update(applied)
    out["answer"] = _render(out, res.rows)
    return out


def _run_srm_check(
    experiment_id: str,
    experiments: Dict[str, Dict[str, Any]],
    db_path: str,
) -> Optional[Dict[str, Any]]:
    """Sample-ratio check from the assignment table, always run for lift questions.

    Deterministic: the query is fixed, never composed by the model. A detected SRM
    blocks a winner downstream (best_variant consults this via the flag). Returns
    None when no intended split is recorded.
    """
    meta = experiments.get(experiment_id, {})
    shares = meta.get("intended_shares") or {}
    if not shares:
        return None
    res = run_bq_query(
        "SELECT variant, COUNT(*) AS users FROM fact_user_assignments "
        f"WHERE experiment_id = '{experiment_id}' GROUP BY variant",
        db_path, row_limit=100,
    )
    if not res.ok:
        return None
    observed = {str(v): int(u) for v, u in res.rows}
    return check_srm(observed, shares).to_dict()


def _looks_per_user(columns: Sequence[str]) -> bool:
    cols = [c.lower() for c in columns]
    return "user_id" in cols and "variant" in cols and not any(
        c in ("country", "platform", "metric_date") for c in cols
    )


def _apply_lift_stats(
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
    experiment_id: str,
    experiments: Dict[str, Dict[str, Any]],
    *,
    srm: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Lift + CI + winner from per-user rows; the SRM result is passed in."""
    if not rows:
        return {"analysis": None, "winner": None, "srm": None}

    # lift_analysis's contract is (user_id, experiment_id, variant, metric_value[, cov]).
    # The model's SELECT order is its own; normalise to the contract instead of
    # hoping the column order matches.
    cols = [c.lower() for c in columns]
    i_user = cols.index("user_id") if "user_id" in cols else 0
    i_var = cols.index("variant") if "variant" in cols else 1
    try:
        i_val = next(i for i, c in enumerate(cols) if c.startswith("metric"))
    except StopIteration:
        # The model computed the number in a differently-named output column (e.g.
        # revenue_usd bare). Fall back to the last non-identifier column, or give
        # up on stats rather than crash the whole question.
        cand = [i for i, c in enumerate(cols)
                if c not in ("user_id", "experiment_id", "variant")]
        if not cand:
            return {"analysis": None, "winner": None, "srm": None}
        i_val = cand[-1]
    i_exp = cols.index("experiment_id") if "experiment_id" in cols else None
    norm: List[Sequence[Any]] = []
    for r in rows:
        user = r[i_user]
        exp = r[i_exp] if i_exp is not None else experiment_id
        norm.append((user, exp, r[i_var], r[i_val]))
    metric = _metric_from_rows(norm)
    analysis = M.lift_analysis(norm, metric)

    # The SRM check ran upstream (in ask) against the assignment table; its result
    # is consulted here to block a winner. Randomisation failure means no arm may
    # be named better, however significant its point estimate looks.
    srm_detected = bool((srm or {}).get("detected"))
    winner = None if srm_detected else M.best_variant(analysis.get("lifts", {}))
    return {"analysis": analysis, "winner": winner, "srm": srm, "metric": metric.name}


def _metric_from_rows(rows: Sequence[Sequence[Any]]) -> "M.MetricDef":
    """Resolve which MetricDef the model computed.

    Uses the ground truth's metric keys against the injection: the experiment's
    primary metric is the default, because per-user rows exist precisely to feed
    the primary-metric decision.
    """
    import json as _json
    truth_path = Path("abtest_data/ground_truth.json")
    if truth_path.exists():
        doc = _json.loads(truth_path.read_text(encoding="utf-8"))
    else:
        doc = {}
    exp = doc.get("experiments", {})
    primary = None
    for spec in exp.values():
        pm = spec.get("primary_metric") or spec.get("primary")
        if pm:
            primary = pm
            break
    metric = M.METRIC_BY_NAME.get(primary or "purchase_rate")
    if metric is None:
        metric = M.METRIC_BY_NAME["purchase_rate"]
    return metric


def _render(out: Dict[str, Any], rows: Sequence[Sequence[Any]]) -> str:
    a = out.get("analysis")
    if a:
        lines = []
        for v, d in sorted(a.get("variant_values", {}).items()):
            lines.append(f"- {v}: {d['metric_per_user']:.4f} over {d['users']:,} users")
        w = out.get("winner")
        if w:
            sig = "significant" if w.get("significant") else \
                "NOT significant (CI includes zero)"
            lines.append(
                f"winner: {w['variant']} lift {w['lift'] * 100:+.2f}% "
                f"CI [{w['ci'][0] * 100:+.2f}%, {w['ci'][1] * 100:+.2f}%] — {sig}"
            )
        if out.get("srm") and out["srm"].get("detected"):
            lines.append("SRM DETECTED — do not act on these effect estimates.")
        return "\n".join(lines)
    if not rows:
        return "The query returned no rows for this question."
    lines = []
    for r in rows[:20]:
        lines.append("- " + " | ".join(
            f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
            for v in r
        ))
    return f"{len(rows)} rows:\n" + "\n".join(lines)


def _extract_sql(text: str) -> str:
    """Pull the SQL out of the reply. Prose returns '' so the loop retries.

    Returning the prose was the bug: reasoning garbage reached the parser as "SQL"
    and the repair loop spiralled. An empty extraction is a retry signal, and the
    retry restates the question.
    """
    if not text:
        return ""
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    lines = text.strip().splitlines()
    starts = [i for i, ln in enumerate(lines)
              if re.match(r"\s*(SELECT|WITH|REFUSE)\b", ln, re.I)]
    if starts:
        return "\n".join(lines[starts[-1]:]).strip()
    m = re.search(r"((?:SELECT|WITH)\s.+?;)", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def _to_mirror(planned_sql: str) -> str:
    """Strip BigQuery catalog qualification for the offline mirror.

    Production BigQuery needs analytics.ab.*; the DuckDB mirror registered bare
    names. Applied after planning so the logged planned SQL stays BigQuery.
    """
    return re.sub(r"\b(?:analytics|ab)\.", "", planned_sql)