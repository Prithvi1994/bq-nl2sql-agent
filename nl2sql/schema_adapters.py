#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Schema introspection for local data files (DuckDB) and BigQuery.

Provides a unified Schema interface that the NL2SQL agent can use
regardless of whether we're querying local files or BigQuery.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import duckdb

from .schema import Schema, Table, Column, ForeignKey


class LocalFileSchema(Schema):
    """
    Schema inferred from local CSV/Parquet/JSON files via DuckDB.
    """
    
    @classmethod
    def from_data_dir(cls, data_dir: str) -> "LocalFileSchema":
        """Create schema by inspecting files in data_dir."""
        if not os.path.isdir(data_dir):
            return cls(tables=[])
        
        conn = duckdb.connect()
        try:
            # Register all files
            import glob
            tables = []
            for filepath in glob.glob(os.path.join(data_dir, "*")):
                if not os.path.isfile(filepath):
                    continue
                table_name = os.path.splitext(os.path.basename(filepath))[0]
                table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
                if not table_name:
                    continue
                
                try:
                    if filepath.endswith(".csv"):
                        conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
                    elif filepath.endswith(".parquet"):
                        conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{filepath}')")
                    elif filepath.endswith(".json") or filepath.endswith(".jsonl"):
                        conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_json_auto('{filepath}')")
                    else:
                        continue
                    
                    # Get schema
                    desc = conn.execute(f"DESCRIBE {table_name}").fetchall()
                    columns = []
                    for row in desc:
                        col_name, col_type, nullable, *_ = row
                        columns.append(Column(
                            name=col_name, 
                            type=col_type.upper(), 
                            notnull=not bool(nullable),
                            pk=False  # We don't know PK from CSV
                        ))
                    
                    # Get row count
                    try:
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    except Exception:
                        row_count = -1
                    
                    tables.append(Table(
                        name=table_name,
                        columns=columns,
                        foreign_keys=[],  # Inferred later via join_hints
                        row_count=row_count
                    ))
                except Exception:
                    # Skip files we can't parse
                    pass
        
        finally:
            conn.close()
        
        return cls(tables=tables)


class BigQuerySchema(Schema):
    """
    Schema from BigQuery INFORMATION_SCHEMA.
    
    Requires google-cloud-bigquery and credentials.
    """
    
    @classmethod
    def from_bigquery(
        cls,
        project: str,
        dataset: str,
        credentials=None,
        location: str = "US",
    ) -> "BigQuerySchema":
        """Create schema by querying BigQuery INFORMATION_SCHEMA."""
        from google.cloud import bigquery
        
        schema = cls()
        client = bigquery.Client(project=project, credentials=credentials, location=location)
        
        # Query INFORMATION_SCHEMA.COLUMNS
        query = f"""
        SELECT 
            table_name,
            column_name,
            data_type,
            is_nullable,
            CASE 
                WHEN column_default IS NOT NULL THEN 'DEFAULT ' || column_default
                ELSE ''
            END as default_expr
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY table_name, ordinal_position
        """
        
        try:
            rows = client.query(query).result()
            for row in rows:
                table_name = row.table_name
                if table_name not in schema.tables:
                    schema.tables[table_name] = TableInfo(name=table_name)
                
                col_type = row.data_type
                if row.default_expr:
                    col_type += f" {row.default_expr}"
                
                schema.tables[table_name].columns.append(ColumnInfo(
                    name=row.column_name,
                    type=col_type,
                    nullable=row.is_nullable == "YES"
                ))
            
            # Get row counts (optional, can be slow for large tables)
            for table_name in schema.tables:
                try:
                    count_query = f"SELECT COUNT(*) FROM `{project}.{dataset}.{table_name}`"
                    row_count = client.query(count_query).result().to_dataframe().iloc[0, 0]
                    schema.tables[table_name].row_count = int(row_count)
                except Exception:
                    schema.tables[table_name].row_count = -1
        
        except Exception as e:
            raise RuntimeError(f"Failed to load BigQuery schema: {e}")
        
        return schema
    
    def join_hints(self, tables: List[str]) -> List[str]:
        """
        Generate join hints using BigQuery's INFORMATION_SCHEMA.TABLE_CONSTRAINTS
        if available, else fall back to column name matching.
        """
        # Try to get actual FK constraints from INFORMATION_SCHEMA
        # For now, fall back to column name matching
        return super().join_hints(tables)


def create_schema(
    backend: str,
    **kwargs
) -> Schema:
    """
    Factory function to create appropriate schema.
    
    Args:
        backend: "local" | "bigquery"
        **kwargs: backend-specific arguments
            local: data_dir
            bigquery: project, dataset, credentials, location
    """
    if backend == "local":
        data_dir = kwargs.get("data_dir", "./data")
        return LocalFileSchema.from_data_dir(data_dir)
    elif backend == "bigquery":
        return BigQuerySchema.from_bigquery(
            project=kwargs["project"],
            dataset=kwargs["dataset"],
            credentials=kwargs.get("credentials"),
            location=kwargs.get("location", "US"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    # Test local schema
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV
        test_csv = os.path.join(tmpdir, "users.csv")
        with open(test_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "value"])
            writer.writerow([1, "alice", 100])
            writer.writerow([2, "bob", 200])
        
        schema = LocalFileSchema.from_data_dir(tmpdir)
        print("Tables:", schema.table_names())
        print("Card:")
        print(schema.card())
        print("Join hints:", schema.join_hints(schema.table_names()))