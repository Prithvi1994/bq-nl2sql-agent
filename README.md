# bq-nl2sql-agent

Agentic **experiment readout platform**: natural-language chat over A/B test
scoresheets, with the semantic layer generated once and the numbers computed
deterministically — the LLM parses intent and narrates, never authoring SQL on
the primary path.

## What Is This

A platform for reading out online controlled experiments from pre-computed
scoresheet tables (one row per experiment × day × variant × slice):

- **Experiment registry** — domain → experiment → physical table; identity
  resolution precedes any query; fail-closed on unknown ids
- **Wren MDL + cube** — the cut catalog: executable metric definitions with
  sum-safety (outcome totals pool across disjoint slices; precomputed rates
  and lifts are read, never aggregated)
- **Deterministic readout** — headline from stored cuts, day-level confidence
  intervals, SRM gate, ramp/segment sweeps, ship/hold decision rule
- **LLM at the edges only** — intent slots in, fixed-order narration out;
  guarded raw-SQL escape hatch behind an sqlglot AST policy gate

## Architecture

```
NL question → intent slots {experiment, metric, slice?, window, grain}
            → registry resolve (the only sanctioned table source)
            → readout() sweep — deterministic dict, no LLM near a number
            → narration (fixed section order, every figure from the dict)
            → agentic chat shell (deepagents; tools discover/resolve/readout)
```

## Quick Start

```bash
pip install -e . && pip install duckdb numpy scipy pyarrow

# Build the fixture dataset + scoresheet layer
python -m nl2sql.abtest.build_dataset --out ./abtest_data

# Register experiments (registry + per-experiment tables)
python -m nl2sql.abtest.register_fixture

# Regression: the readout must recover the known ground truth (26/26)
python -m nl2sql.abtest.verify
```

## Layout

```
nl2sql/
├── bq_exec.py              # BigQuery-dialect executor over the DuckDB mirror
└── abtest/
    ├── build_dataset.py    # statistically-honest 3-experiment fixture + scoresheet
    ├── register_fixture.py # registry seeding (the onboarding path)
    ├── registry.py         # domain → experiment → table, fail-closed
    ├── policy.py           # sqlglot-AST SQL gate (escape hatch + slice rules)
    ├── stats.py            # SRM, lift, Wilson CI, Welch t, CUPED, decision rule
    ├── readout.py          # readout sweep: headline/ramp/segments/gates
    ├── verify.py           # effect-recovery regression suite
    └── wren_project/       # 1 MDL model + 1 cube + population rules
```

## Eval philosophy

Ground truth is the injected truth, computed independently of the code under
test; refusals score harsher than answers; every historical bug is a permanent
test case; floors are reported, not means.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
