from nl2sql.llm import create_llm
from nl2sql.agent import NL2SQLAgent
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb
from nl2sql.evaluate import load_golden, GoldenCase, CaseOutcome, EvalReport
from collections import Counter
import time
import statistics

def _norm_value(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip().casefold()
        try:
            return round(float(s), 2)
        except ValueError:
            return s
    return v

def normalise_rows(rows):
    return Counter(tuple(_norm_value(v) for v in r) for r in rows)

def rows_match(candidate, reference):
    return normalise_rows(candidate) == normalise_rows(reference)

schema = LocalFileSchema.from_data_dir('./data')

def duckdb_executor(db_path, sql, row_limit=200, timeout_s=30.0):
    return run_query_duckdb(sql, data_dir='./data', row_limit=row_limit, timeout_s=timeout_s)

golden_cases = load_golden('golden/local_test.json')
golden_map = {g.question: g.sql for g in golden_cases}

class GoldenMockLLM:
    def __init__(self, golden_map):
        self.golden_map = golden_map
    
    def complete(self, messages):
        user_content = messages[-1]['content']
        for q, sql in self.golden_map.items():
            if q in user_content:
                return f"```sql\n{sql}\n```"
        return "```sql\nSELECT 1\n```"

agent = NL2SQLAgent(
    db_path=':memory:',
    llm=GoldenMockLLM(golden_map),
    schema=schema,
    executor=lambda db, sql, **kw: duckdb_executor(None, sql, **kw),
)

# Custom evaluation using DuckDB for both reference and candidate
def evaluate_duckdb(agent, golden_cases):
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    cases_out = []
    
    for g in golden_cases:
        # Run reference with DuckDB
        ref = duckdb_executor(None, g.sql, row_limit=agent.row_limit, timeout_s=agent.timeout_s)
        if not ref.ok:
            raise ValueError(f"Golden case {g.id} reference SQL failed: {ref.error}")
        
        res = agent.ask(g.question)
        exec_ok = res.status != "failed"
        match = exec_ok and rows_match(res.rows, ref.rows)
        
        last = res.attempts[-1]
        error = "; ".join(last.guard_errors) or last.exec_error
        
        cases_out.append(CaseOutcome(
            id=g.id, question=g.question, status=res.status, exec_ok=exec_ok, match=match,
            judged=None, repairs=res.repairs_used, confidence=res.confidence,
            total_ms=res.total_ms, candidate_sql=res.sql, reference_sql=g.sql,
            candidate_rows=len(res.rows), reference_rows=len(ref.rows), error=error, tags=g.tags
        ))
    
    return EvalReport(llm="mock:golden", cases=cases_out, started=started,
                      elapsed_s=time.perf_counter() - t0)

report = evaluate_duckdb(agent, golden_cases)
print(report.to_markdown())
print()
print("Summary:", report.summary())