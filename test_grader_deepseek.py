#!/usr/bin/env python
import os
import sys

# Ensure OPENROUTER_API_KEY is set
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# Use deepseek for grading (fast, free)
from nl2sql.report_graph import run_report_generation, ReportStyle
from nl2sql.llm import create_llm
from nl2sql.prompts import grade_report, GraderResult
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb
from nl2sql.agent import NL2SQLAgent
from nl2sql.examples import ExampleBank
from nl2sql.prompts import _format_data_for_llm

# Test the grader with deepseek
print("Testing grader with deepseek-v4-flash...")

# Setup
schema = LocalFileSchema.from_data_dir('./data')
agent = NL2SQLAgent(
    db_path=':memory:',
    llm=create_llm('nemotron-3-ultra'),  # Use nemotron for SQL gen
    schema=schema,
    executor=lambda db, sql, **kw: __import__('nl2sql.executor_duckdb').executor_duckdb.run_query_duckdb(sql, data_dir='./data', **kw),
)

# Get a test question and generate report
question = "Total revenue by country"
result = agent.ask(question)

# Grade the report
grader_llm = create_llm("deepseek/deepseek-v4-flash-0731:free")

# Mock report for testing
report = f"""## Executive Summary

Total revenue across all countries is **$3,169.49**, with Germany leading at $1,200.00 (37.9%).

## Key Metrics

| Country | Revenue | Share |
|---------|---------|-------|
| Germany | $1,200.00 | 37.9% |
| UK | $664.50 | 21.0% |
| USA | $534.99 | 16.9% |
| Canada | $450.00 | 14.2% |
| France | $320.00 | 10.1% |

## Methodology

Query joined customers and orders tables, filtered for completed orders, grouped by country.
"""

# Run grader
print("Running grader with deepseek-v4-flash...")
result = grade_report(
    question="Total revenue by country",
    report=report,
    sql="SELECT c.country, SUM(o.total_amount) as revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'completed' GROUP BY c.country ORDER BY revenue DESC",
    columns=["country", "revenue"],
    rows=[("Germany", 1200.0), ("UK", 664.5), ("USA", 534.99), ("Canada", 450.0), ("France", 320.0)],
    style="executive_summary",
    model="deepseek/deepseek-v4-flash-0731:free",
)

print(f"Decision: {result.decision}")
print(f"Scores: {result.scores}")
print(f"Feedback: {result.feedback}")
print(f"Overall: {result.overall}")