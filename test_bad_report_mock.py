from nl2sql.llm import LLM, Messages
from typing import List, Dict, Callable, Sequence, Union
import re

class BadReportMockLLM:
    """Mock LLM that generates SQL correctly but produces a BAD report for testing REVISE."""
    
    def __init__(self):
        from nl2sql.llm import OracleLLM
        self.oracle = OracleLLM("golden/local_test.json")
        self.call_count = 0
    
    @property
    def name(self):
        return "mock:bad_report"
    
    def complete(self, messages: Messages) -> str:
        self.call_count += 1
        user = messages[-1]['content'].rstrip()
        
        # Check if it's a SQL generation prompt
        if 'Schema:' in user and 'Question:' in user:
            return self._get_sql_from_oracle(user)
        
        # Check if it's a report generation prompt
        if 'Data (' in user and 'Style:' in user:
            return self._generate_bad_report()
        
        # Check if it's a revision prompt
        if 'Revise the report' in user:
            return self._generate_good_report()  # On revision, generate good report
        
        # Check if it's a judge prompt
        if 'Two SQL queries' in user or 'CANDIDATE' in user:
            return 'YES\nThe candidate answers the question correctly.'
        
        return '```sql\nSELECT 1\n```'
    
    def _get_sql_from_oracle(self, user: str) -> str:
        import re
        match = re.search(r'Question:\s*(.+)$', user, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip().rstrip('?')
            norm = extracted.lower().replace('what is ', '').replace('what are ', '').strip()
            for key, sql in self.oracle.normalized_answers.items():
                if key in norm or norm in key:
                    return f'```sql\n{sql}\n```'
        return '```sql\nSELECT 1\n```'
    
    def _generate_bad_report(self) -> str:
        """Generate a BAD report with hallucinations and errors."""
        return '''## Executive Summary

Total revenue across all countries is **$5,000,000** from completed orders.

## Revenue by Country

| Country | Revenue |
|---------|---------|
| Germany | $2,000,000 |
| UK | $1,500,000 |
| Canada | $1,000,000 |
| USA | $500,000 |

## Key Insights

- Germany leads with $2M (40% of total)
- France is second at $1.5M (30%)
- Japan has the lowest revenue at $100K (2%)

## Methodology

Query joined customers, orders, products, and inventory tables, filtered for all orders including cancelled ones, grouped by country and product category.

*Data based on 10,000 orders across 15 countries.'''
    
    def _generate_good_report(self) -> str:
        return '''## Executive Summary

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

*Data based on 4 completed orders across 4 countries.'''