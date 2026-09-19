# bq-nl2sql-agent

**Fork of [nl2sql-agent](https://github.com/gandhi1994/nl2sql-agent) by [gandhi1994](https://github.com/gandhi1994) (35k+ ⭐)**

BigQuery-compatible Natural Language to SQL agent with **grounded evaluation**, **local testing via DuckDB**, and **production-ready BigQuery backend**.

## What's New in This Fork

**Enhanced with:**
- **LangGraph-based report generation pipeline** with auto-grader eval loop
- **Automated grader/eval loop** — groundedness, faithfulness, factual_precision, completeness, style adherence
- **Report styles**: executive summary, detailed analysis, technical brief, slide deck
- **Automated grader/eval loop** replacing human-in-the-loop
- **32+ new golden test cases** covering all question categories
- **Comprehensive eval framework** (Text-to-SQL + Report Generation)

---

## Quick Start (Local Testing)

```bash
# Clone and install
git clone https://github.com/your-org/bq-nl2sql-agent.git
cd bq-nl2sql-agent
pip install -e .

# Or just install dependencies
pip install duckdb sqlglot openai pydantic python-dotenv

# Run a question against the sample data
python -m nl2sql.cli_local --question "Total revenue by country"

# Or with custom data
python -m nl2sql.cli_local --question "Top 5 customers" --data-dir ./my_data --model gpt-4o
```

## Sample Data

The repo includes sample CSV files in `./data/`:
- `customers.csv` - 8 customers
- `orders.csv` - 10 orders
- `products.csv` - 10 products
- `order_items.csv` - 13 line items

## What Questions Can You Ask?

The agent handles a wide range of analytical questions. Here are categories with examples:

### **Aggregations & Metrics**
```bash
python -m nl2sql.cli_local -q "Total revenue by country"
python -m nl2sql.cli_local -q "Average order value by country"
python -m nl2sql.cli_local -q "Total revenue by product category"
python -m nl2sql.cli_local -q "Number of orders per month"
python -m nl2sql.cli_local -q "Average items per order"
```

### **Top-N & Rankings**
```bash
python -m nl2sql.cli_local -q "Which customer spent the most?"
python -m nl2sql.cli_local -q "Top 5 products by revenue"
python -m nl2sql.cli_local -q "Best selling category"
python -m nl2sql.cli_local -q "Most frequent customers"
```

### **Filtering & Segmentation**
```bash
python -m nl2sql.cli_local -q "List all electronics products with their prices"
python -m nl2sql.cli_local -q "Orders over $500 in the last 30 days"
python -m nl2sql.cli_local -q "Customers from USA who ordered electronics"
python -m nl2sql.cli_local -q "Pending orders from UK customers"
```

### **Joins & Relationships**
```bash
python -m nl2sql.cli_local -q "What products were in order 1001?"
python -m nl2sql.cli_local -q "Which customers bought laptops?"
python -m nl2sql.cli_local -q "Products never ordered"
python -m nl2sql.cli_local -q "Customer order history with product names"
```

### **Complex Analytics**
```bash
python -m nl2sql.cli_local -q "Revenue by country and category"
python -m nl2sql.cli_local -q "Customer lifetime value"
python -m nl2sql.cli_local -q "Month-over-month revenue growth"
python -m nl2sql.cli_local -q "Repeat purchase rate by country"
```

### **Set Operations & Existence**
```bash
python -m nl2sql.cli_local -q "Customers who haven't placed any orders"
python -m nl2sql.cli_local -q "Products that were never sold"
python -m nl2sql.cli_local -q "Countries with no completed orders"
```

### **Date/Time Analysis** (if your data has dates)
```bash
python -m nl2sql.cli_local -q "Orders in June 2023"
python -m nl2sql.cli_local -q "Revenue trend over time"
python -m nl2sql.cli_local -q "First order date per customer"
```

## Configuration: API Keys

The agent auto-detects your provider from environment variables:

| Provider | Environment Variable | Example Models |
|----------|---------------------|----------------|
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-haiku` |
| **OpenRouter** | `OPENROUTER_API_KEY` | `nemotron-3-ultra`, `pareto-code`, `deepseek-v3`, `llama-3.1-70b`, `qwen-2.5-coder`, `gemma-2-9b` |

```bash
# Set your key
export OPENAI_API_KEY="sk-..."           # OpenAI
export ANTHROPIC_API_KEY="sk-ant-..."    # Anthropic (via OpenRouter)
export OPENROUTER_API_KEY="sk-or-..."    # OpenRouter

# Run with auto-detection
python -m nl2sql.cli_local -q "Total revenue by country"

# Or specify model explicitly
python -m nl2sql.cli_local -q "Top customers" --model gpt-4o-mini
python -m nl2sql.cli_local -q "Top customers" --model claude-3-5-sonnet
python -m nl2sql.cli_local -q "Top customers" --model nemotron-3-ultra
```

## Using Your Own Data

Place CSV, Parquet, or JSON files in a directory:

```
my_data/
├── customers.csv
├── orders.csv
├── products.csv
└── order_items.csv
```

```bash
python -m nl2sql.cli_local --question "Revenue by customer" --data-dir ./my_data
```

**Supported formats:**
- `.csv` — Auto-detects types, headers
- `.parquet` — Preserves types exactly
- `.json` / `.jsonl` — Line-delimited or array

## Running the Evaluation Suite

The evaluation harness scores on three metrics:

1. **Execution Success** — SQL ran without error
2. **Result Match** — Candidate rows == Reference rows (order-insensitive, normalized)
3. **Judge Rescue** — LLM judge says "candidate answers question as well as reference" (used only when result match fails)

### Run Local Evaluation (DuckDB)

```bash
# Using the test script with golden test cases
python test_eval.py
```

Expected output:
```
# NL2SQL evaluation: mock:golden

| Metric                    | Value  |
|---------------------------|--------|
| Execution success         | 100%   |
| Result match (deterministic)| 100%  |
| Judge-rescued cases       | 0      |
| Accepted (match or judged)| 100%   |
| Median latency per query  | 33 ms  |
```

### Run Evaluation with Real LLM

```bash
# Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# Run evaluation (uses OpenRouter for judge rescue)
python -c "
from nl2sql.evaluate import load_golden, evaluate
from nl2sql.agent import NL2SQLAgent
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb
from nl2sql.llm import create_llm

schema = LocalFileSchema.from_data_dir('./data')
golden = load_golden('golden/local_test.json')

llm = create_llm('nemotron-3-ultra')
agent = NL2SQLAgent(
    db_path=':memory:',
    llm=llm,
    schema=schema,
    executor=lambda db, sql, **kw: run_query_duckdb(sql, data_dir='./data', **kw),
)

report = evaluate(agent, golden, judge=llm)
print(report.to_markdown())
"
```

### Custom Golden Test Cases

Add your own test cases to `golden/local_test.json`:

```json
[
  {
    "id": "custom-1",
    "question": "Your business question",
    "sql": "SELECT ... FROM ... WHERE ...",
    "tags": ["join", "aggregation"]
  }
]
```

Then run:
```bash
python test_eval.py
```

## Report Generation Pipeline

The fork includes a **LangGraph-based report generation pipeline** with auto-grader eval loop:

```bash
# Generate a report
python -c "
from nl2sql.report_graph import run_report_generation, ReportStyle

report = run_report_generation(
    question='Total revenue by country',
    style=ReportStyle.EXECUTIVE,
    model='gpt-4o-mini',
    data_dir='./data',
)
print(report['summary'])
"
```

### Report Styles

| Style | Sections | Use Case |
|-------|----------|----------|
| `executive_summary` | 2-3 | Leadership updates, KPI dashboards |
| `detailed_analysis` | 4-6 | Deep dives, segment breakdowns |
| `technical_brief` | 3-4 | Reproducibility, query logic, data quality |
| `slide_deck` | 5-8 slides | Presentations, stakeholder meetings |

### Auto-Grader Eval Loop

The pipeline includes an **automated grader** that evaluates:
- **Groundedness** — Every claim supported by source data
- **Faithfulness** — No contradictions with source data
- **Factual Precision** — Numbers match source exactly
- **Completeness** — Covers totals, trends, outliers, methodology
- **Style Adherence** — Matches requested format

The grader replaces human-in-the-loop with automated evaluation:
```
generate_report → auto_grader → [APPROVE → finalize | REVISE → revise_report → auto_grader (loop) | REJECT → handle_error]
```

## Production: BigQuery Backend

```bash
# Install BigQuery extras
pip install -e ".[bigquery]"

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

# Use BigQuery schema and executor
from nl2sql.schema_adapters import BigQuerySchema
from nl2sql.executor_bigquery import run_query_bigquery

schema = BigQuerySchema.from_bigquery(project="my-proj", dataset="my_dataset")
agent = NL2SQLAgent(..., schema=schema, executor=run_query_bigquery)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      NL2SQLAgent                             │
├─────────────────────────────────────────────────────────────┤
│  Schema (LocalFileSchema | BigQuerySchema)                  │
│  Guard (sqlglot dialect: sqlite/bigquery)                   │
│  Executor (DuckDB local | BigQuery remote)                  │
│  LLM (OpenAI-compatible: OpenAI, Anthropic, OpenRouter)     │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| File | Purpose |
|------|---------|
| `agent.py` | Main agent loop: generate → guard → execute → repair |
| `guard.py` | SQL validation via sqlglot (dialect-aware) |
| `schema_adapters.py` | Schema introspection (local files / BigQuery) |
| `executor_duckdb.py` | Local execution on CSV/Parquet via DuckDB |
| `executor_bigquery.py` | Production BigQuery execution (optional) |
| `evaluate.py` | Grounded eval harness (execution + result match + judge) |
| `report_graph.py` | LangGraph report pipeline with auto-grader |
| `prompts.py` | Generation/repair/judge/grader prompts |
| `cli_local.py` | Zero-setup local testing CLI |

## Attribution

This project is a fork of **nl2sql-agent** by **gandhi1994** (original: https://github.com/gandhi1994/nl2sql-agent, 35k+ ⭐).

**Original work:** © gandhi1994 (MIT License)  
**Enhancements:** © [Your Name/Org] (MIT License)

See [NOTICE](NOTICE) for full attribution details.

## License

MIT License — see [LICENSE](LICENSE) for details.