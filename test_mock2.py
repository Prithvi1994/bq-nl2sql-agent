from nl2sql.llm import create_llm
from nl2sql.agent import NL2SQLAgent
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb

schema = LocalFileSchema.from_data_dir('./data')

def duckdb_executor(db_path, sql, row_limit=200, timeout_s=30.0):
    return run_query_duckdb(sql, data_dir='./data', row_limit=row_limit, timeout_s=timeout_s)

# Test question 2: Which customer spent the most?
agent = NL2SQLAgent(
    db_path=':memory:',
    llm=create_llm('mock', replies=[
        '''```sql
SELECT c.first_name, c.last_name, SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC LIMIT 1
```'''
    ]),
    schema=schema,
    executor=duckdb_executor,
)

result = agent.ask('Which customer spent the most?')
print('Q: Which customer spent the most?')
print('Status:', result.status)
print('SQL:', result.sql)
print('Rows:', result.rows)
print()

# Test question 3: List all electronics products
agent2 = NL2SQLAgent(
    db_path=':memory:',
    llm=create_llm('mock', replies=[
        '''```sql
SELECT name, price FROM products WHERE category = 'Electronics' ORDER BY price DESC
```'''
    ]),
    schema=schema,
    executor=duckdb_executor,
)

result2 = agent2.ask('List all electronics products with their prices')
print('Q: List all electronics products with their prices')
print('Status:', result2.status)
print('SQL:', result2.sql)
print('Rows:', result2.rows)
print()

# Test question 4: How many orders per customer
agent3 = NL2SQLAgent(
    db_path=':memory:',
    llm=create_llm('mock', replies=[
        '''```sql
SELECT c.first_name, c.last_name, COUNT(o.order_id) as order_count
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY order_count DESC
```'''
    ]),
    schema=schema,
    executor=duckdb_executor,
)

result3 = agent3.ask('How many orders per customer?')
print('Q: How many orders per customer?')
print('Status:', result3.status)
print('SQL:', result3.sql)
print('Rows:', result3.rows)