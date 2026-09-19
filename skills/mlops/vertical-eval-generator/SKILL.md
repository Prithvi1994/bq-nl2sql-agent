---
name: vertical-eval-generator
description: Generate vertical golden evals from schema using Spider 2.0.
category: mlops
tags: [evaluation, nl2sql, golden-suite, vertical, spider2]
---

# Vertical Eval Generator Skill

## Purpose
Generate domain-specific (vertical) golden evaluation suites for NL2SQL agents based on user's actual data schema and business patterns. Uses Spider 2.0 methodology adapted for vertical use cases.

## Trigger
Use when user wants to build a golden evaluation suite for their specific domain/schema (e-commerce, fintech, healthcare, etc.) rather than generic benchmarks.

## Workflow

### Phase 1: Schema Discovery & Profiling
1. **Connect to data source** (BigQuery, local files, Postgres, etc.)
2. **Extract schema** - tables, columns, types, FK relationships
3. **Profile data** - run distinct values, null rates, cardinality, value ranges
4. **Infer semantics** - detect dimensions, measures, time columns, categorical values

### Phase 2: Domain Pattern Elicitation
Ask clarifying questions to understand:
- **Business entities** (customers, orders, products, etc.)
- **Key metrics** (revenue, LTV, churn, conversion)
- **Time granularity** (daily, fiscal quarters, business days)
- **Segmentation dimensions** (geography, channel, cohort)
- **Data quality rules** (soft deletes, currency, timezone)

### Phase 3: Query Pattern Mining
Generate query templates across categories:
- **KPI/Metrics** (aggregations, trends)
- **Cohort Analysis** (retention, LTV by cohort)
- **Funnel/Conversion** (multi-step flows)
- **Segment Comparison** (A vs B analysis)
- **Time Series** (MoM, YoY, rolling windows)
- **Anomaly Detection** (outliers, threshold breaches)
- **Operational** (pending, inventory, exceptions)

### Phase 4: Golden Case Generation
For each pattern:
1. **Generate canonical SQL** using schema + business logic
2. **Create natural language question** variations
3. **Add metadata**: category, difficulty, tags, expected rows
4. **Validate** - execute SQL, verify results make sense
5. **Export** to `golden/vertical.json`

## Key Files

```
vertical_eval_generator/
├── skill.py                    # Main entry point
├── schema_profiler.py         # Schema + data profiling
├── pattern_library.py         # Query pattern templates
├── question_generator.py      # NL question templates
├── sql_generator.py           # Canonical SQL generator
├── validator.py               # Execute + verify
├── cli.py                     # CLI interface
└── templates/
    ├── ecommerce_patterns.yaml
    ├── fintech_patterns.yaml
    └── healthcare_patterns.yaml
```

## Usage

```bash
# Interactive mode - asks clarifying questions
python -m vertical_eval_generator --interactive

# From BigQuery
python -m vertical_eval_generator --bq-project my-proj --bq-dataset my_dataset

# From local files
python -m vertical_eval_generator --data-dir ./data --domain ecommerce

# Output
# → golden/vertical.json (30-40 cases)
# → golden/business_logic.py (quarter defs, currency, etc.)
# → golden/data_quality_rules.py
```

## Clarifying Questions (Examples)

### For Each Table:
1. What business entity does this represent?
2. What is the primary key?
3. Which columns are dimensions vs measures?
4. Are there soft-delete flags? (e.g., `deleted_at`, `is_active`)
5. What timezone are timestamps in?
6. Any currency columns? Which currency?

### For Relationships:
1. What are the FK relationships? (explicit or implicit)
2. Are there many-to-many tables?
3. Any slowly-changing dimensions (SCD Type 2)?

### For Business Logic:
1. How do you define fiscal quarters?
2. What constitutes "active" customer?
3. How is LTV calculated?
4. What is "churn" definition?
5. Any custom aggregations?

## Data Profiling Commands

```sql
-- Distinct values for categorical columns
SELECT column_name, COUNT(DISTINCT column_name) as cardinality
FROM table_name
GROUP BY column_name;

-- Null rates
SELECT column_name, COUNT(*) - COUNT(column_name) as null_count
FROM table_name;

-- Value ranges for numeric
SELECT column_name, MIN(column_name), MAX(column_name), AVG(column_name)
FROM table_name;

-- Time granularity
SELECT DATE_TRUNC(created_at, DAY) as day, COUNT(*)
FROM table_name
GROUP BY day
ORDER BY day DESC LIMIT 30;
```

## Spider 2.0 Adaptation for Vertical

| Spider Concept | Vertical Equivalent |
|----------------|---------------------|
| Cross-domain schemas | **Your production schema** |
| 10K generic questions | **30-40 domain-specific patterns** |
| Easy/Medium/Hard/Extra-hard | **Your difficulty tiers** (simple KPI → complex cohort) |
| Execution Accuracy | **Business Logic Accuracy** |
| External knowledge (fiscal quarters) | **Your business definitions** |

## Output Format

```json
[
  {
    "id": "kpi-001",
    "category": "kpi_metrics",
    "question": "Total revenue by country last quarter",
    "sql": "SELECT c.country, SUM(o.total_amount) FROM ... WHERE o.order_date >= '2024-01-01' AND o.order_date < '2024-04-01' GROUP BY c.country ORDER BY revenue DESC",
    "difficulty": "easy",
    "business_logic": "fiscal_quarter_definition",
    "data_quality": "handles_null_country",
    "expected_rows": 5,
    "tags": ["date_filter", "join", "aggregation", "quarterly"],
    "variations": [
      "Revenue by country for Q1 2024",
      "Show me country-wise revenue last quarter"
    ]
  }
]
```

## Integration with NL2SQL Agent

```python
# In eval harness
from vertical_eval_generator import load_vertical_golden

golden = load_vertical_golden("golden/vertical.json")
report = evaluate(agent, golden, judge=llm)
print(report.to_markdown())
```

## Requirements
- Python 3.10+
- duckdb (for local profiling)
- google-cloud-bigquery (for BQ)
- pyyaml, pandas