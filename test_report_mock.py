from nl2sql.llm import LLM, Messages
from typing import List, Dict, Callable, Sequence, Union
import re


class ReportTestMockLLM:
    """Mock LLM that handles both SQL generation (via Oracle) and report generation."""

    def __init__(self):
        from nl2sql.llm import OracleLLM
        self.oracle = OracleLLM("golden/local_test.json")

    @property
    def name(self):
        return "mock:test"

    def complete(self, messages: Messages) -> str:
        user = messages[-1]["content"].rstrip()

        # Check if it's a SQL generation prompt (contains 'Schema:' and 'Question:')
        if "Schema:" in user and "Question:" in user:
            return self._get_sql_from_oracle(user)

        # Check if it's a report generation prompt (contains 'Data (' and 'Style:')
        if "Data (" in user and "Style:" in user:
            return self._generate_mock_report()

        # Check if it's a revision prompt
        if "Revise the report" in user:
            return self._generate_mock_report()

        # Check if it's a judge prompt
        if "Two SQL queries" in user or "CANDIDATE" in user:
            return "YES\nThe candidate answers the question correctly."

        return "```sql\nSELECT 1\n```"

    def _get_sql_from_oracle(self, user: str) -> str:
        import re
        # Try to extract question
        match = re.search(r"Question:\s*(.+)$", user, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip().rstrip("?")
            norm = extracted.lower().replace("what is ", "").replace("what are ", "").strip()
            # Use oracle's normalized answers
            # We need access to oracle - let's create one
            from nl2sql.llm import OracleLLM
            oracle = OracleLLM("golden/local_test.json")
            for key, sql in oracle.normalized_answers.items():
                if key in norm or norm in key:
                    return f"```sql\n{sql}\n```"
        return "```sql\nSELECT 1\n```"

    def _generate_mock_report(self) -> str:
        return """## Executive Summary

Total revenue across all countries is **$2,614.49** from completed orders.

## Revenue by Country

| Country | Revenue |
|---------|---------|
| Germany | $1,200.00 |
| UK | $639.50 |
| Canada | $450.00 |
| USA | $324.99 |

## Key Insights

- Germany leads with $1,200.00 (46% of total)
- UK is second at $639.50 (24%)
- USA has the lowest revenue at $324.99 (12%)

## Methodology

Query joined customers and orders tables, filtered for completed orders only, grouped by country.

*Data based on 4 completed orders across 4 countries."""