"""Agentic experiment-readout skill — the 5-tool wrapper over the deterministic core.

Tools: discover_experiments / resolve_experiment / profile_cuts / readout / report.
Contract: skill = the ONLY LLM-touching layer; every number in any output is
mechanically traceable to the readout dict (hash-stable). The skill receives NL
questions, maps them to tool calls (slot-fill via the runtime agent), and grades
its own narration with the report-layer faithfulness check.

Run grounding eval:  python -m nl2sql.abtest.eval_skill
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from nl2sql.abtest.profile import profile_cuts
from nl2sql.abtest.registry import RegistryError, experiments_for_domain, resolve
from nl2sql.abtest.sweep import sweep
from nl2sql.abtest.report import build_report

DB_DEFAULT = "./abtest_data/abtest.duckdb"


# --------------------------------------------------------------------------- #
# Tool 1 — discovery
# --------------------------------------------------------------------------- #
def discover_experiments(db: str = DB_DEFAULT, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    if domain:
        validate_identifier(domain)
        return experiments_for_domain(db, domain)
    # no domain given: union of all registry domains
    import duckdb
    con = duckdb.connect(db, read_only=True)
    doms = [r[0] for r in con.execute(
        "SELECT DISTINCT domain FROM dim_experiment_registry ORDER BY 1").fetchall()]
    con.close()
    out: List[Dict[str, Any]] = []
    for d in doms:
        out.extend(experiments_for_domain(db, d))
    return out


# --------------------------------------------------------------------------- #
# Tool 2 — resolution (fail closed)
# --------------------------------------------------------------------------- #
def resolve_experiment(domain: str, experiment_id: str, db: str = DB_DEFAULT) -> Dict[str, Any]:
    validate_identifier(domain)
    validate_identifier(experiment_id)
    return resolve(db, domain, experiment_id)


# --------------------------------------------------------------------------- #
# Tool 3 — zoom-candidate profile
# --------------------------------------------------------------------------- #
def profile_cuts_tool(db: str, domain: str, experiment_id: str,
                      min_cell: int = 1000) -> Dict[str, Any]:
    row = resolve_experiment(domain, experiment_id, db)
    return profile_cuts(db, experiment_id, row, min_cell=min_cell)


# --------------------------------------------------------------------------- #
# Tool 4 — readout (sweep + verdict), zoom-honouring
# --------------------------------------------------------------------------- #
def readout(db: str, domain: str, experiment_id: str, *,
            dims: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    row = resolve_experiment(domain, experiment_id, db)
    prof = profile_cuts(db, experiment_id, row)
    # honour tiers: only request exploratory+precommitted+Overall values that
    # passed cell floor; diagnostic-dim values are excluded from the sweep inputs
    safe = {"slice_country": [], "slice_page": []}
    for tier_name in ("precommitted", "exploratory"):
        for entry in prof["tiers"][tier_name] if isinstance(prof["tiers"][tier_name], list) else []:
            dim = entry["dim"]
            if entry.get("cell_ok"):
                safe.setdefault(dim, []).append(entry["value"])
    if dims:  # caller intent narrows, never broadens beyond safe set
        for k, v in dims.items():
            have = set(safe.get(k, []))
            safe[k] = [x for x in v if x in have or x == "Overall"]
    return sweep(db, experiment_id, row, slices=safe).to_dict()


# --------------------------------------------------------------------------- #
# Tool 5 — report (HTML + PDF with narration + faithfulness gate)
# --------------------------------------------------------------------------- #
def report_tool(db: str, domain: str, experiment_id: str, out_path: str,
                fmt: str = "pdf") -> str:
    return str(build_report(db, experiment_id, domain, out_path, pdf=(fmt == "pdf")))


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def validate_identifier(s: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]+", s or ""):
        raise RegistryError(f"invalid identifier: {s!r}")
    return s


TOOL_SPECS = [
    {"name": "discover_experiments",
     "desc": "List experiments (optionally one domain). Returns registry rows: id, domain, window, north star, ship threshold.",
     "params": {"domain": "str?"}},
    {"name": "resolve_experiment",
     "desc": "Verify an experiment exists in a domain; return its table, variants, window, north-star metric. Fails closed on unknowns.",
     "params": {"domain": "str", "experiment_id": "str"}},
    {"name": "profile_cuts",
     "desc": "Introspect the experiment's scoresheet: dimension columns, cell sizes, zoom tiers (decisional/precommitted/exploratory/diagnostic), suspect-dim guard.",
     "params": {"domain": "str", "experiment_id": "str", "min_cell": "int?"}},
    {"name": "readout",
     "desc": "Run the deterministic lift sweep + go/no-go verdict. Accepts optional dim filter; never broadens past cell-size floor or suspect-dim guard.",
     "params": {"domain": "str", "experiment_id": "str", "dims": "{dim:[values]}?"}},
    {"name": "report",
     "desc": "Produce the readout document (PDF/HTML) — charts, deterministic tables, LLM narration gated by number-level faithfulness vs the readout dict.",
     "params": {"domain": "str", "experiment_id": "str", "out_path": "str", "fmt": "'pdf'|'html'"}},
]
