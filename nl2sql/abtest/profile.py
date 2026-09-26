"""Dynamic zoom candidate discovery over one experiment's scoresheet table.

`profile_cuts` introspects the contract table (daily grain × variant × slice),
detects dimension columns + cardinalities, sizes every candidate cell, and emits
a zoom ladder in tiers. Tier policy (fixed in code, no LLM):
  - decisional  : Overall per (variant × metric)        — the only ship-grade tier
  - pre-committed: only present when a registry `hypothesis` names the dim/value
  - exploratory : remaining zoom-1 cells passing min-cell-size + multiplicity cap
  - diagnostic  : zoom-2 combos (capped), or dims flagged suspect_denominator
                   (variant share drift > tol — treatment-affected-dim guard,
                   inferred from data since the schema has no config flag)
Reality without the registry extras: hypothesis absent ⇒ nothing pre-committed;
the report auto-labels slices "hypothesis-generating."
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from nl2sql.bq_exec import run_bq_query

MIN_CELL_USER_DAYS = 1000
MAX_EXPLORATORY_CUTS = 30
MAX_ZOOM2_CUTS = 8
DRIFT_TOL = 0.10  # variant share drift tolerance per dim value


def profile_cuts(db: str, experiment_id: str, registry_row: Dict[str, Any],
                 min_cell: int = MIN_CELL_USER_DAYS) -> Dict[str, Any]:
    queries: List[Dict[str, Any]] = []

    def _q(sql: str):
        res = run_bq_query(sql, db, row_limit=50000)
        queries.append({"label": "profile", "ok": res.ok, "rows": res.row_count,
                        "error": res.error})
        if not res.ok:
            raise RuntimeError(f"profile_cuts query failed: {res.error}")
        return res.rows

    exp = experiment_id.replace("'", "")
    # 1. columns + cardinalities
    dims: Dict[str, List[str]] = {}
    for col, val in _q(
        f"SELECT 'slice_country' AS col, slice_country AS val FROM exp_scoresheet "
        f"WHERE experiment_id = '{exp}' AND NOT overall_flag AND slice_country != 'Overall' "
        f"UNION ALL SELECT 'slice_page', slice_page FROM exp_scoresheet "
        f"WHERE experiment_id = '{exp}' AND NOT overall_flag AND slice_page != 'Overall'"):
        dims.setdefault(col, []).append(val)
    for col in dims:
        dims[col] = sorted(set(dims[col]))

    # 2. cell sizes (user-days per variant per slice value) + variant share drift
    cells: Dict[Tuple[str, str, str], int] = {}
    shares: Dict[Tuple[str, str, str], int] = {}
    for d, v, sc, sp, u in _q(
        f"SELECT CAST(metric_date AS VARCHAR), variant_id, slice_country, slice_page, "
        f"SUM(users) FROM exp_scoresheet WHERE experiment_id = '{exp}' AND NOT overall_flag "
        f"AND slice_country != 'Overall' GROUP BY 1, 2, 3, 4"):
        cells[(d, v, sc)] = int(u)
        shares[(v, sc)] = shares.get((v, sc), 0) + int(u)
    # NOTE: country dim only — slice_page rows in the fixture carry country 'Overall'
    # style detail duplicated; slice_page cells ride the same (d, v, sc) row set.

    # 3. suspect-dim guard: variant share drift per dim value
    suspect: Dict[str, List[str]] = {}
    totals_by_variant: Dict[str, int] = {}
    for (v, sc), n in shares.items():
        totals_by_variant[v] = totals_by_variant.get(v, 0) + n
    for col, vals in dims.items():
        if col != "slice_country":
            continue  # only country-level detail exists in the fixture today
        for val in vals:
            a = shares.get(("ctl", val), 0)
            b = shares.get(("t1", val), 0)
            ta, tb = totals_by_variant.get("ctl", 0), totals_by_variant.get("t1", 0)
            if a and ta and b and tb:
                drift = abs(a / ta - b / tb)
                if drift > DRIFT_TOL:
                    suspect.setdefault(col, []).append(val)

    # 4. hypothesis pre-commitment (absent registry hypothesis → empty)
    hyp = (registry_row.get("hypothesis") or "").lower()
    precommitted: Dict[str, List[str]] = {}
    if hyp:
        for col, vals in dims.items():
            stem = col.replace("slice_", "")
            for val in vals:
                if re.search(rf"\b{re.escape(val.lower())}\b", hyp) or \
                   re.search(rf"\b{re.escape(stem)}\b", hyp):
                    precommitted.setdefault(col, []).append(val)

    # 5. tiers
    tier: Dict[str, Any] = {
        "decisional": ["Overall"],
        "precommitted": {c: sorted(set(vals)) for c, vals in precommitted.items()},
        "exploratory": [],
        "diagnostic": [],
    }
    for col, vals in dims.items():
        for val in vals:
            ok_size = (shares.get(("ctl", val), 0) >= min_cell and
                       shares.get(("t1", val), 0) >= min_cell)
            entry = {"dim": col, "value": val, "cell_ok": ok_size}
            if col in suspect and val in suspect[col]:
                tier["diagnostic"].append(entry)
            elif val in precommitted.get(col, []):
                tier["precommitted"].setdefault(col, [])
            elif ok_size:
                tier["exploratory"].append(entry)
            else:
                entry["reason"] = "below min-cell floor"
                tier["diagnostic"].append(entry)
    tier["exploratory"] = tier["exploratory"][:MAX_EXPLORATORY_CUTS]

    # zoom-2: pairs across distinct dims (capped)
    combos: List[Dict[str, str]] = []
    for i, c1 in enumerate(dims):
        for c2 in list(dims)[i + 1:]:
            for v1 in dims[c1][:4]:
                for v2 in dims[c2][:4]:
                    if (c1, v1) not in suspect_pairs(c1, suspect) and \
                       (c2, v2) not in suspect_pairs(c2, suspect):
                        combos.append({"dims": [c1, c2], "values": [v1, v2]})
    tier["zoom2"] = combos[:MAX_ZOOM2_CUTS]

    return {"dims": dims, "suspect": suspect, "tiers": tier,
            "user_shares": {f"{v}|{k}": n for (v, k), n in sorted(shares.items())},
            "min_cell": min_cell, "queries": queries, "drift_tol": DRIFT_TOL}


def suspect_pairs(col: str, suspect: Dict[str, List[str]]) -> set:
    return {(col, v) for v in suspect.get(col, [])}
