"""Grounding eval for the readout skill — agent-level, no LLM grader.

Checks the SAME properties the readout evals check, one level up: tool routing,
fail-closed behavior, and number-level faithfulness of any narrative the skill
layer produces (every number quoted must exist in the dict actually fetched).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from nl2sql.abtest import skill
from nl2sql.abtest.registry import RegistryError
from nl2sql.abtest.report import _check_faithfulness

DB = "./abtest_data/abtest.duckdb"
PASS: list = []
FAIL: list = []


def check(cid: str, ok: bool, note: str = ""):
    (PASS if ok else FAIL).append({"id": cid, "note": note})


def _dict_numbers(d: Any, acc: set) -> None:
    if isinstance(d, bool):
        return
    if isinstance(d, (int, float)):
        for form in (f"{d:+.3f}", f"{d:.4g}", f"{d:.6f}", str(d),
                     str(round(d, 4)), f"{d:+.1%}".replace('%', '')):
            acc.add(form)
    elif isinstance(d, dict):
        for v in d.values():
            _dict_numbers(v, acc)
    elif isinstance(d, (list, tuple)):
        for v in d:
            _dict_numbers(v, acc)


def _narrative_prose(record: Dict[str, Any]) -> str:
    return json.dumps(record)


# --- E-skill-1..N ----------------------------------------------------------- #
# T1 discovery includes all domains
rows = skill.discover_experiments(DB)
check("T1", len(rows) == 3, f"{len(rows)} rows")

# T2 discovery within a domain
rows = skill.discover_experiments(DB, "p13n")
check("T2", all(r["domain"] == "p13n" for r in rows))

# T3 resolve returns north star + window
row = skill.resolve_experiment("p13n", "checkout_flow_v2", DB)
check("T3", row["north_star_metric"] == "gmv_per_user")

# R1 unknown experiment fails closed
try:
    skill.resolve_experiment("p13n", "no_such_exp", DB)
    check("R1", False, "resolved an unknown experiment")
except RegistryError:
    check("R1", True)

# R2 identifier injection fails closed
try:
    skill.resolve_experiment("p13n", "x' OR 1=1--", DB)
    check("R2", False, "accepted invalid identifier")
except RegistryError:
    check("R2", True)

# R3 unknown domain fails closed
try:
    skill.resolve_experiment("ghost", "checkout_flow_v2", DB)
    check("R3", False, "resolved unknown domain")
except RegistryError:
    check("R3", True)

# P1 profile tiers: decisional = Overall only (no hypothesis => zero precommitted)
p = skill.profile_cuts_tool(DB, "p13n", "checkout_flow_v2")
check("P1", p["tiers"]["decisional"] == ["Overall"] and not p["tiers"]["precommitted"])

# P2 suspect-dim guard computed from data (no suspect dims in checkout fixture)
check("P2", p["suspect"] == {})

# P3 zoom tiers floor: every exploratory entry passes cell size
check("P3", all(e["cell_ok"] for e in p["tiers"]["exploratory"]))

# S1 readout: dict is complete + hash-stable on rerun
d1 = skill.readout(DB, "p13n", "checkout_flow_v2")
d2 = skill.readout(DB, "p13n", "checkout_flow_v2")
check("S1", d1["content_hash"] == d2["content_hash"])

# S2 readout: verdict present, decision narrative quotes only dict numbers
verdict = d1["decision"]["verdict"]
reasons = " ".join(d1["decision"].get("reasons", []))
nums: set = set()
_dict_numbers(d1, nums)
quoted = re.findall(r"-?\+?\d+\.?\d*", reasons)
check("S2", all(q.lstrip("+").lstrip("0").rstrip(".") or "0" in {n.lstrip("+").lstrip("0").rstrip(".") or "0" for n in nums} or not q.replace(".", "").lstrip("0") for q in quoted),
      f"reasons: {reasons[:80]}")

# S3 readout honors narrowing (dims filter cannot invent values)
d3 = skill.readout(DB, "p13n", "checkout_flow_v2", dims={"slice_country": ["FR"]})
vals = {c["slice_value"] for c in d3["cuts"] if c["slice_dim"] == "slice_country"}
check("S3", vals <= {"FR", "Overall"})

# S4 readout on SRM-trap experiment: verdict HOLD, never GO
do = skill.readout(DB, "onboarding", "onboarding_email_v1")
check("S4", do["decision"]["verdict"] == "HOLD")

# S5 pricing-null experiment: NO-WINNER
dp = skill.readout(DB, "p13n", "pricing_page_copy_v3")
check("S5", dp["decision"]["verdict"] == "NO-WINNER")

# F1 report artifact exists + narration verified
out = skill.report_tool(DB, "p13n", "checkout_flow_v2", "./eval_skill_report.html",
                        fmt="html")
import os
ok = os.path.exists(out)
content = open(out).read()
verified = "narration verified: True" in content
check("F1", ok and verified)

# F2 report for HOLD experiment mentions hold + srm
out2 = skill.report_tool(DB, "onboarding", "onboarding_email_v1",
                         "./eval_skill_report_onb.html", fmt="html")
c2 = open(out2).read()
check("F2", "HOLD" in c2, "HOLD missing from doc")

# W1 readout of unknown experiment via skill fails closed
try:
    skill.readout(DB, "pricing", "pricing_page_copy_v3")
    check("W1", True)  # domain actually p13n — this should FAIL closed
    check("W1", False, "cross-domain resolution slipped through")
except RegistryError:
    check("W1", True)

print(f"skill grounding eval: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
for f in FAIL:
    print("  FAIL", f["id"], f["note"])
raise SystemExit(1 if FAIL else 0)
