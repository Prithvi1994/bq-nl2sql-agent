from nl2sql.llm import create_llm
from nl2sql.agent import NL2SQLAgent
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb

# Setup
schema = LocalFileSchema.from_data_dir('./data')
llm = create_llm('mock', replies=[
    '''```sql
SELECT c.country, SUM(o.total_amount) as revenue 
FROM customers c 
JOIN orders o ON c.customer_id = o.customer_id 
WHERE o.status = 'completed' 
GROUP BY c.country 
ORDER BY revenue DESC
```'''
])

def duckdb_executor(db_path, sql, row_limit=200, timeout_s=30.0):
    return run_query_duckdb(sql, data_dir='./data', row_limit=row_limit, timeout_s=timeout_s)

agent = NL2SQLAgent(
    db_path=':memory:',
    llm=llm,
    schema=schema,
    executor=duckdb_executor,
)

result = agent.ask('Total revenue by country')
print('Status:', result.status)
print('SQL:', result.sql)
print('Columns:', result.columns)
print('Rows:', result.rows)
print('Attempts:', len(result.attempts))
for a in result.attempts:
    print(f'  Attempt: kind={a.kind}, sql={a.sql}, guard_errors={a.guard_errors}, exec_error={a.exec_error}')