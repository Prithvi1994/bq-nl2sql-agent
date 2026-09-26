# bq-nl2sql-agent

Agentic **experiment readout platform**: one command turns a daily-grain
experiment scoresheet into a grounded go/no-go readout — PDF report with
charts, verified LLM narration, and every number traceable to a hash-stable
deterministic dict. The LLM parses intent and narrates; it never authoring
SQL on the primary path and never computes a number.

## What Is This

A platform for reading out online controlled experiments from pre-computed
scoresheet tables (one row per experiment × day × variant × slice):

- **Experiment registry** — domain → experiment → physical table; identity
  resolution precedes any query; fail-closed on unknown ids
- **Deterministic readout sweep** — pooled lift CIs (sum-ratio / day-cluster),
  weekly ramp stability, SRM gate (HOLD overrides any signal), guardrail
  checks, go/no-go verdict, content hash for reproducibility
- **Zoom ladder over cuts** — dynamic dimension discovery + cell-size floors:
  *decisional* (Overall) / *precommitted* (hypothesis text, if registered) /
  *exploratory* (filtered, multiplicity-capped) / *diagnostic* (suspect-dim
  share drift is inferred from the data)
- **Readout skill (5 tools)** — `discover / resolve / profile / readout /
  report` as a repo-level skill (`skill_cli`); any new experiment conforming
  to the schema is onboardable with a registry row, no code changes
- **Report generation** — matplotlib charts + Jinja HTML → weasyprint PDF;
  the LLM narration passes a number-faithfulness gate (every quoted figure
  must exist in the dict) or falls back to template prose; the PDF always ships
- **Wren MDL + cube** — the governed metric catalog: executable measures with
  sum-safety (totals pool across disjoint slices; precomputed rates and lifts
  are read, never re-aggregated); guarded raw-SQL escape hatch behind an
  sqlglot AST policy gate
- **LLM at the edges only**

## Readout skill — one command

```bash
python -m nl2sql.abtest.skill_cli discover
python -m nl2sql.abtest.skill_cli readout --domain p13n --exp checkout_flow_v2
python -m nl2sql.abtest.skill_cli readout --domain p13n --exp checkout_flow_v2 \
    --dims "slice_country=US,FR" --json readout_dict.json
python -m nl2sql.abtest.skill_cli report --domain p13n --exp checkout_flow_v2 \
    --out ./readout_checkout          # -> readout_checkout.pdf
```

Sample output (the shipped 3-experiment fixture):

```
experiment: checkout_flow_v2
verdict: NO-WINNER
SRM: clean (p=0.863035)        ramp: stable=False
n cuts tested: 42              dict sha256: ff46b90c0573e1ae
  SIG  t1  atc_per_user  US  lift=+12.5% ci=[+9.7%,+15.4%]
  SIG  t1  atc_per_user  FR  lift=+8.7%  ci=[+3.8%,+13.7%]
  SIG  t2  atc_per_user  BR  lift=-11.6% ci=[-16.6%,-6.3%]   ...
```

Onboarding (`onboarding_email_v1`) carries a planted SRM trap (62/38 vs
intended 50/50): the sweep returns **HOLD** and refuses a winner even though
t1 shows significant per-country lifts.

## Architecture

```
NL question / one command
   → discover + resolve (registry; the only sanctioned table source)
   → profile_cuts (dynamic dims, zoom tiers, cell floors, suspect-dim guard)
   → readout() sweep — deterministic dict (sha256-stable), no LLM near a number
   → report: charts + tables (deterministic) + narration (LLM, faithfulness-gated)
   → PDF / JSON dict / CLI prose — every figure traceable to the dict
```

## Layout

```
nl2sql/
├── bq_exec.py              # BigQuery-dialect executor over the DuckDB mirror
└── abtest/
    ├── build_dataset.py    # statistically-honest 3-experiment fixture + scoresheet
    ├── register_fixture.py # registry seeding (the onboarding path)
    ├── registry.py         # domain → experiment → table, fail-closed
    ├── profile.py          # zoom-candidate discovery: dims, tiers, floors, drift guard
    ├── skill.py            # 5-tool readout skill wrapper
    ├── skill_cli.py        # one-command CLI entrypoints (discover/readout/report)
    ├── sweep.py            # deterministic cut sweep: lift CIs, ramp, SRM, verdict
    ├── charts.py           # ramp + slice-lift bar charts (matplotlib)
    ├── report.py           # HTML/PDF assembly + faithfulness-gated narration
    ├── policy.py           # sqlglot-AST SQL gate (escape hatch + slice rules)
    ├── stats.py            # SRM, Wilson CI, Welch t, CUPED, decision rule
    ├── eval_readout.py     # 22 truth/trap/refusal cases on the sweep
    ├── eval_skill.py       # 17 agent-level grounding checks
    ├── verify.py           # 26 effect-recovery regressions
    └── wren_project/       # 1 MDL model + 1 cube + population rules
```

## Eval gates (all green — CI-able)

| Suite | Cases | What it proves |
|---|---|---|
| `python -m nl2sql.abtest.verify` | 26/26 | fixture truths recovered (effect/SRM/decision) |
| `python -m nl2sql.abtest.eval_readout` | 22/22 | traps: Overall+detail mixing (the live 2× double-count), dollar recompute, wrong slice, window sensitivity, denominator; refusals fail closed |
| `python -m nl2sql.abtest.eval_skill` | 17/17 | agent-level grounding: routing, narrowing invariant, hash stability, verdict routing, report verification |

Ground truth is the injected truth, computed independently of the code under
test; refusals score harsher than answers; every historical bug is a permanent
test case.

## Fixture (what ships)

3 experiments × 30k assigned users × up to 56 test days (+ pre-period), 7
country slices, north star `gmv_per_user`: a winning effect, a null effect,
and an SRM-trap hold — so all three verdict classes are exercisable offline.
The agent-facing surface is `exp_scoresheet` (≈2.6k rows) + the registry;
per-user truth tables exist only to verify the fixture.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
