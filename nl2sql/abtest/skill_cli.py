"""Repo-level readout skill: one-command entrypoints for any caller (human, agent, CI).

Usage:
  python -m nl2sql.abtest.skill_cli discover [--domain p13n]
  python -m nl2sql.abtest.skill_cli readout --domain p13n --exp checkout_flow_v2
  python -m nl2sql.abtest.skill_cli readout --domain p13n --exp checkout_flow_v2 --json out.json
  python -m nl2sql.abtest.skill_cli profile --domain p13n --exp checkout_flow_v2
  python -m nl2sql.abtest.skill_cli report --domain p13n --exp checkout_flow_v2 --out ./my_readout

All commands read only the scoresheet + registry; no table creation; fail closed.
The `report` command is the full readout synthesis: deterministic tables +
charts + LLM prose (faithfulness-gated) -> PDF, ready to share.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from nl2sql.abtest.registry import RegistryError
from nl2sql.abtest.skill import (DB_DEFAULT,
                                  discover_experiments,
                                  profile_cuts_tool,
                                  readout,
                                  report_tool,
                                  resolve_experiment)


def _parse_dims(spec: Optional[str]) -> Optional[Dict[str, List[str]]]:
    if not spec:
        return None
    out: Dict[str, List[str]] = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        key, _, vals = part.partition("=")
        out[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    return out


def _print_readout(d: Dict[str, Any]) -> None:
    dd = d["decision"]
    print(f"experiment: {d['experiment_id']}")
    print(f"verdict: {dd['verdict']}" + (f" -> {dd['winner']}" if dd.get("winner") else ""))
    for r in dd.get("reasons", []):
        print(f"  - {r}")
    srm = d.get("srm", {})
    if srm.get("detected") is not None:
        print(f"SRM: {'DETECTED' if srm['detected'] else 'clean'}"
              + (f" (p={srm['p_value']:.6f})" if srm.get("p_value") is not None else ""))
    ramp = d.get("ramp", {})
    if ramp:
        print(f"ramp: stable={ramp.get('stable')}")
    print(f"n cuts tested: {dd.get('n_tests', len(d.get('cuts', [])))}")
    print(f"dict sha256: {d['content_hash']}")
    for c in d["cuts"]:
        lift = c.get("lift")
        if lift is None or not c.get("significant"):
            continue
        ci = c.get("ci") or [None, None]
        loc = f"{c['slice_dim']}={c['slice_value']}" if c["slice_value"] != "Overall" else "Overall"
        print(f"  SIG {c['variant_id']:>3} {c['metric']:>14} {loc:>18} "
              f"lift={lift:+.1%} ci=[{ci[0]:+.1%},{ci[1]:+.1%}] p={c['p']:.3f}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="readout", description="Experiment readout skill")
    ap.add_argument("--db", default=DB_DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discover")
    p_disc.add_argument("--domain", default=None)

    p_res = sub.add_parser("resolve")
    p_res.add_argument("--domain", required=True)
    p_res.add_argument("--exp", required=True)

    p_prof = sub.add_parser("profile")
    p_prof.add_argument("--domain", required=True)
    p_prof.add_argument("--exp", required=True)
    p_prof.add_argument("--min-cell", type=int, default=1000)

    p_read = sub.add_parser("readout")
    p_read.add_argument("--domain", required=True)
    p_read.add_argument("--exp", required=True)
    p_read.add_argument("--dims", default=None, help="slice_country=US,FR;slice_page=pdp")
    p_read.add_argument("--json", default=None, help="write full dict to file")

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--domain", required=True)
    p_rep.add_argument("--exp", required=True)
    p_rep.add_argument("--out", default=None, help="output path base (no ext)")
    p_rep.add_argument("--fmt", choices=["pdf", "html"], default="pdf")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "discover":
            rows = discover_experiments(args.db, args.domain)
            for r in rows:
                print(f"{r['domain']:>12}/{r['experiment_id']:<24} "
                      f"north_star={r.get('north_star_metric')}")
            return 0
        if args.cmd == "resolve":
            row = resolve_experiment(args.domain, args.exp, args.db)
            print(json.dumps({k: row[k] for k in row if k != "queries"},
                             indent=1, default=str))
            return 0
        if args.cmd == "profile":
            p = profile_cuts_tool(args.db, args.domain, args.exp,
                                  min_cell=args.min_cell)
            print(json.dumps(p, indent=1, default=str))
            return 0
        if args.cmd == "readout":
            d = readout(args.db, args.domain, args.exp, dims=_parse_dims(args.dims))
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(d, f, indent=1, default=str)
                print(f"dict written: {args.json} (sha {d['content_hash'][:16]})")
            _print_readout(d)
            return 0
        if args.cmd == "report":
            base = args.out or f"./readout_{args.exp}"
            out = report_tool(args.db, args.domain, args.exp, base,
                              fmt=args.fmt)
            print(f"report: {out}" + (".pdf" if args.fmt == "pdf" else ""))
            return 0
    except RegistryError as e:
        print(f"refused (fail closed): {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
