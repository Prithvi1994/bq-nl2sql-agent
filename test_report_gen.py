#!/usr/bin/env python
import os
import sys

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# Use a faster model for report generation
from nl2sql.report_graph import run_report_generation, ReportStyle

report = run_report_generation(
    question='Total revenue by country',
    style='executive_summary',
    model='nemotron-3-ultra',
    data_dir='./data',
    enable_human_review=False,
)

import json
print('Summary:', report['summary'][:200] if report.get('summary') else 'None')
print('---')
for section in report.get('sections', []):
    print(f'## {section["title"]}')
    print(section['content'][:200])
    print('---')