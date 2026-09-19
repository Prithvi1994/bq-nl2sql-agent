#!/usr/bin/env python
"""
Schema Profiler - Extracts schema and profiles data for vertical eval generation.
Supports: BigQuery, local files (CSV/Parquet), DuckDB, Postgres.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import duckdb
import pandas as pd


@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    nullable: bool
    distinct_count: int
    null_count: int
    null_rate: float
    sample_values: List[Any]
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    avg_val: Optional[float] = None
    is_pk: bool = False
    is_fk: bool = False
    fk_table: Optional[str] = None
    fk_column: Optional[str] = None
    semantic_type: Optional[str] = None  # dimension, measure, time, categorical, id


@dataclass
class TableProfile:
    """Profile of a single table."""
    name: str
    row_count: int
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    pk_columns: List[str] = field(default_factory=list)
    fk_relationships: List[Dict[str, str]] = field(default_factory=list)
    business_entity: Optional[str] = None  # e.g., "customer", "order", "product"


@dataclass
class SchemaProfile:
    """Complete schema profile."""
    tables: Dict[str, TableProfile] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    domain: Optional[str] = None  # e.g., "ecommerce", "fintech"


class SchemaProfiler:
    """Profiles database schema and data."""

    def __init__(self, data_source: str, source_type: Literal["bq", "local", "duckdb", "postgres"] = "local"):
        self.data_source = data_source
        self.source_type = source_type
        self.profile = SchemaProfile()

    def profile_local_files(self, data_dir: str = ".") -> SchemaProfile:
        """Profile local CSV/Parquet files using DuckDB."""
        conn = duckdb.connect()
        data_path = Path(data_dir)
        
        for file_path in data_path.glob("*.csv"):
            table_name = file_path.stem
            self._profile_table(conn, table_name, str(file_path))
        
        for file_path in data_path.glob("*.parquet"):
            table_name = file_path.stem
            self._profile_table(conn, table_name, str(file_path))
        
        self._infer_relationships()
        return self.profile

    def _profile_table(self, conn, table_name: str, file_path: str):
        """Profile a single table."""
        # Get schema
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{file_path}') LIMIT 0").fetchdf()
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{file_path}')").fetchone()[0]
        
        table = TableProfile(name=table_name, row_count=row_count)
        
        for _, col_row in schema_df.iterrows():
            col_name = col_row['column_name']
            col_type = col_row['column_type']
            
            # Profile column
            col_profile = self._profile_column(conn, table_name, file_path, col_name, col_type)
            table.columns[col_name] = col_profile
            
            # Detect PK
            if col_profile.distinct_count == row_count and row_count > 0:
                col_profile.is_pk = True
                table.pk_columns.append(col_name)
        
        self.profile.tables[table_name] = table

    def _profile_column(self, conn, table_name: str, file_path: str, col_name: str, col_type: str) -> ColumnProfile:
        """Profile a single column."""
        # Distinct count
        distinct_result = conn.execute(
            f"SELECT COUNT(DISTINCT {col_name}) FROM read_csv_auto('{file_path}')"
        ).fetchone()
        distinct_count = distinct_result[0] if distinct_result else 0
        
        # Null count
        null_result = conn.execute(
            f"SELECT COUNT(*) - COUNT({col_name}) FROM read_csv_auto('{file_path}')"
        ).fetchone()
        null_count = null_result[0] if null_result else 0
        
        # Sample values
        sample_result = conn.execute(
            f"SELECT DISTINCT {col_name} FROM read_csv_auto('{file_path}') LIMIT 10"
        ).fetchall()
        sample_values = [r[0] for r in sample_result]
        
        # Min/max/avg for numeric
        min_val = max_val = avg_val = None
        if any(t in col_type.upper() for t in ['INT', 'FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC']):
            stats = conn.execute(
                f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM read_csv_auto('{file_path}')"
            ).fetchone()
            if stats:
                min_val, max_val, avg_val = stats
        
        return ColumnProfile(
            name=col_name,
            dtype=col_type,
            nullable=null_count > 0,
            distinct_count=distinct_count,
            null_count=null_count,
            null_rate=null_count / max(1, conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{file_path}')").fetchone()[0]),
            sample_values=sample_values,
            min_val=min_val,
            max_val=max_val,
            avg_val=avg_val,
        )

    def _infer_relationships(self):
        """Infer FK relationships from column names."""
        tables = self.profile.tables
        for table_name, table in tables.items():
            for col_name, col in table.columns.items():
                # Check if column matches PK of another table
                for other_table_name, other_table in tables.items():
                    if other_table_name == table_name:
                        continue
                    if col_name in other_table.pk_columns:
                        col.is_fk = True
                        col.fk_table = other_table_name
                        col.fk_column = col_name
                        table.fk_relationships.append({
                            "column": col_name,
                            "ref_table": other_table_name,
                            "ref_column": col_name,
                        })
                        self.profile.relationships.append({
                            "from_table": table_name,
                            "from_column": col_name,
                            "to_table": other_table_name,
                            "to_column": col_name,
                        })

    def infer_semantic_types(self, business_context: Optional[Dict] = None):
        """Infer semantic types (dimension, measure, time, etc.) from data and context."""
        for table_name, table in self.profile.tables.items():
            for col_name, col in table.columns.items():
                col.semantic_type = self._infer_column_semantic(col_name, col, table, business_context)

    def _infer_column_semantic(self, col_name: str, col: ColumnProfile, table: TableProfile, 
                                business_context: Optional[Dict]) -> str:
        """Infer semantic type of a column."""
        name_lower = col_name.lower()
        
        # ID columns
        if col.is_pk or name_lower.endswith('_id') or name_lower == 'id':
            return 'id'
        
        # Time columns
        if any(t in name_lower for t in ['date', 'time', 'created', 'updated', 'timestamp']):
            return 'time'
        
        # Currency/amount
        if any(t in name_lower for t in ['amount', 'price', 'cost', 'revenue', 'total', 'sum', 'value']):
            return 'measure'
        
        # Count/quantity
        if any(t in name_lower for t in ['count', 'qty', 'quantity', 'number']):
            return 'measure'
        
        # Categorical - low cardinality relative to rows
        if table.row_count > 0 and col.distinct_count / table.row_count < 0.05 and col.distinct_count < 50:
            return 'categorical'
        
        # High cardinality text - likely dimension
        if col.dtype in ('VARCHAR', 'TEXT', 'STRING') and col.distinct_count > 10:
            return 'dimension'
        
        # Default
        return 'dimension' if col.dtype in ('VARCHAR', 'TEXT', 'STRING') else 'measure'

    def to_dict(self) -> Dict:
        """Convert profile to dictionary."""
        return {
            "tables": {
                name: {
                    "name": t.name,
                    "row_count": t.row_count,
                    "business_entity": t.business_entity,
                    "pk_columns": t.pk_columns,
                    "fk_relationships": t.fk_relationships,
                    "columns": {
                        c_name: {
                            "name": c.name,
                            "dtype": c.dtype,
                            "nullable": c.nullable,
                            "distinct_count": c.distinct_count,
                            "null_count": c.null_count,
                            "null_rate": c.null_rate,
                            "sample_values": c.sample_values,
                            "min_val": c.min_val,
                            "max_val": c.max_val,
                            "avg_val": c.avg_val,
                            "is_pk": c.is_pk,
                            "is_fk": c.is_fk,
                            "fk_table": c.fk_table,
                            "fk_column": c.fk_column,
                            "semantic_type": c.semantic_type,
                        }
                        for c_name, c in t.columns.items()
                    }
                }
                for name, t in self.profile.tables.items()
            },
            "relationships": self.profile.relationships,
            "domain": self.profile.domain,
        }

    def save(self, output_path: str):
        """Save profile to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def profile_data_dir(data_dir: str, domain: Optional[str] = None) -> SchemaProfile:
    """Convenience function to profile a data directory."""
    profiler = SchemaProfiler(data_source=data_dir, source_type="local")
    profile = profiler.profile_local_files(data_dir)
    profile.domain = domain
    profiler.infer_semantic_types()
    return profile


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile data directory for vertical eval generation")
    parser.add_argument("--data-dir", default=".", help="Directory with CSV/Parquet files")
    parser.add_argument("--domain", help="Domain name (ecommerce, fintech, etc.)")
    parser.add_argument("--output", default="schema_profile.json", help="Output JSON file")
    args = parser.parse_args()
    
    profile = profile_data_dir(args.data_dir, args.domain)
    profile.save(args.output)
    print(f"Profile saved to {args.output}")
    print(f"Tables: {list(profile.tables.keys())}")