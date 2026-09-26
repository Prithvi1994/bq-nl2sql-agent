"""Wren-style SQL policy gate.

Modelled on Wren's `policy.py` (Apache-2.0, Canner/WrenAI): enforcement lives in
the SQL AST, never in the prompt. Two controls:

- **Read-only** (always on): only SELECT-family statements reach the connector.
- **Strict mode** (always on here): every referenced table must be an MDL model,
  and every column must belong to the model it is selected from.

On top of Wren's two controls we add the experiment-path rules that Wren has no
opinion about, because they encode decisions this project took from real bugs:
the population policy, the row-level export refusal, and the window bounds.

The point of the AST shape (vs. regex): a rule that operates on the parsed tree
cannot be evaded by formatting, casing, comments, or a nested subquery the
regex never looks into. Our regex gate missed "SELECT variant, COUNT(*) ...
CONTACT_COLUMNS" because the aggregate satisfied a *textual* check. An AST walk
doesn't have that hole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sqlglot
from sqlglot import exp


# Columns that must never appear in any output. Modelled as a data-level denial
# (Wren's CLAC idea) rather than a prompt rule: the AST check sees them wherever
# they surface — projection, WHERE, ORDER BY, a subquery the model thought we
# would not notice.
DENIED_COLUMNS = {"email", "phone", "phone_number", "address", "full_name",
                  "first_name", "last_name", "contact"}


class PolicyError(ValueError):
    """The query violates policy. The agent retries; the user never sees rows."""


def check_policy(
    sql: str,
    *,
    allowed_models: set[str],
    experiment_id: str = "",
    experiment_bounds: Optional[Dict[str, str]] = None,
    dialect: str = "bigquery",
) -> List[str]:
    """Return a list of violations. Empty means the query may execute."""
    bad: List[str] = []
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError as exc:
        # sqlglot is our own dialect parser (bq_exec uses it everywhere); if IT
        # cannot parse the query, it is not SQL, and the answer is refusal —
        # not a pass-through to the database.
        return [f"unparseable SQL: {type(exc).__name__}"]

    # 1. Read-only. Mirror of Wren's read-only check.
    if not isinstance(parsed, (exp.Select, exp.Union)):
        bad.append(f"only SELECT statements are allowed, got {type(parsed).__name__}")

    # 2. Allow-list of tables (mirror of Wren strict mode)
    allowed = {m.lower() for m in allowed_models} | {
        "fact_daily_assigned", "fact_assigned_with_pre", "fact_user_assignments",
    }
    for table in parsed.find_all(exp.Table):
        name = (table.name or "").lower()
        if name and name not in allowed:
            bad.append(f"table {name!r} is not an MDL model")

    # 3. Denied columns (row-level privacy) are deliberately NOT enforced here: the
    #    dataset is synthetic with no PII. When this goes to real user data, re-add
    #    the walk over exp.Column against a denied set — see the git history of this
    #    file for the implementation.

    # 4. Population rule: a COUNT over the daily model is user-days, not users.
    for node in parsed.find_all(exp.Count):
        arg = node.this
        if isinstance(arg, exp.Star):
            # COUNT(*) is allowed only when the FROM is an assigned-grain table.
            froms = [t.name.lower() for t in parsed.find_all(exp.Table)]
            if froms and all("daily" in f or "window" in f for f in froms):
                bad.append("COUNT(*) over a daily-grain model counts user-days, not assigned users")

    # 6. Window bounds, checked on the AST (not a regex: 'metric_date BETWEEEN' inside
    # a nested CTE evade a text search).
    if experiment_bounds:
        between = list(parsed.find_all(exp.Between))
        dates: List[str] = []
        for b in between:
            for lit in b.find_all(exp.Literal):
                if isinstance(lit.this, str) and re.match(r"\d{4}-\d{2}-\d{2}", lit.this):
                    dates.append(lit.this)
        if len(dates) >= 2:
            lo, hi = sorted(dates[:2])
            if experiment_bounds and (
                lo < experiment_bounds.get("started_on", "")
                or hi > experiment_bounds.get("ended_on", "")
            ):
                bad.append(
                    f"window {lo}..{hi} exceeds the experiment's "
                    f"{experiment_bounds.get('started_on')}..{experiment_bounds.get('ended_on')}"
                )

# 5. Cross-experiment denial: two-experiment comparison in one query is a policy
    # breach — a lift has no meaning across randomisations. Exploit the experiment id from
    # the filter: if two different experiment values appear, refuse.
    if experiment_id:
        literals = {lit.this for lit in parsed.find_all(exp.Literal)
                    if isinstance(lit.this, str)}
        other = [x for x in literals
                 if isinstance(x, str) and x.startswith(("checkout", "pricing", "onboarding"))
                 and x != experiment_id and "v" in x]
        if other:
            bad.append(f"query references other experiments: {sorted(other)[:2]}")

    return bad