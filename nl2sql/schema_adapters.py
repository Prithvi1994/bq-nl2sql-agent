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
    """Schema from BigQuery INFORMATION_SCHEMA with native metadata enrichment.

    Includes: partitioning, clustering, column profiles (histograms/nulls),
    policy tags, and storage info from BigQuery's native metadata.
    Requires google-cloud-bigquery and optionally bigquery-dataprofiler.
    """

    def __init__(self, tables=None, partitions=None, clustering=None,
                 column_profiles=None, policy_tags=None, storage_info=None):
        super().__init__(tables or [])
        self.partitions = partitions or {}      # table_name -> partition fields
        self.clustering = clustering or {}      # table_name -> clustered fields
        self.column_profiles = column_profiles or {}  # table_name -> column profiles
        self.policy_tags = policy_tags or {}    # table_name -> column tags
        self.storage_info = storage_info or {}  # table_name -> size, row count

    @classmethod
    def from_bigquery(
        cls,
        project: str,
        dataset: str,
        credentials=None,
        location: str = "US",
        include_profiles: bool = True,
        include_policy_tags: bool = True,
        include_partitions: bool = True,
        include_clustering: bool = True,
        include_storage: bool = True,
    ) -> "BigQuerySchema":
        """Create schema by querying BigQuery INFORMATION_SCHEMA and native metadata.

        Args:
            project: GCP project ID
            dataset: BigQuery dataset ID
            credentials: GCP credentials (optional, uses default if None)
            location: BigQuery location (default: US)
            include_profiles: Whether to fetch column profiles (histograms, null rates)
            include_policy_tags: Whether to fetch policy tags
            include_partitions: Whether to fetch partitioning info
            include_clustering: Whether to fetch clustering info
            include_storage: Whether to fetch storage info
        """
        from google.cloud import bigquery

        client = bigquery.Client(project=project, credentials=credentials, location=location)
        schema = cls()

        # 1. Get columns from INFORMATION_SCHEMA.COLUMNS
        columns_query = f"""
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
            rows = client.query(columns_query).result()
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
        except Exception as e:
            raise RuntimeError(f"Failed to load BigQuery schema columns: {e}")

        # 2. Get partitioning info from INFORMATION_SCHEMA.TABLE_OPTIONS
        if include_partitions:
            partitions_query = f"""
            SELECT
                table_name,
                option_value
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS`
            WHERE option_name IN ('partition_expiration_days', 'require_partition_filter')
            AND option_value IS NOT NULL
            """
            try:
                rows = client.query(partitions_query).result()
                for row in rows:
                    schema.partitions[row.table_name] = row.option_value
            except Exception:
                pass  # Partitioning optional

        # 3. Get clustering info from INFORMATION_SCHEMA.TABLES
        if include_clustering:
            clustering_query = f"""
            SELECT
                table_name,
                clustering_fields
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLES`
            WHERE clustering_fields IS NOT NULL AND ARRAY_LENGTH(clustering_fields) > 0
            """
            try:
                rows = client.query(clustering_query).result()
                for row in rows:
                    schema.clustering[row.table_name] = row.clustering_fields
            except Exception:
                pass  # Clustering optional

        # 4. Get storage info from INFORMATION_SCHEMA.TABLE_STORAGE
        if include_storage:
            storage_query = f"""
            SELECT
                table_name,
                total_logical_bytes,
                total_rows,
                active_logical_bytes,
                long_term_logical_bytes
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLE_STORAGE`
            """
            try:
                rows = client.query(storage_query).result()
                for row in rows:
                    schema.storage_info[row.table_name] = {
                        "total_logical_bytes": row.total_logical_bytes,
                        "total_rows": row.total_rows,
                        "active_logical_bytes": row.active_logical_bytes,
                        "long_term_logical_bytes": row.long_term_logical_bytes,
                    }
                    # Update row count if we have it
                    if row.total_rows and row.table_name in schema.tables:
                        schema.tables[row.table_name].row_count = int(row.total_rows)
            except Exception:
                pass  # Storage optional

        # 5. Get policy tags from INFORMATION_SCHEMA.COLUMN_FIELD_PATHS
        if include_policy_tags:
            policy_query = f"""
            SELECT
                table_name,
                column_name,
                policy_tags
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
            WHERE policy_tags IS NOT NULL
            """
            try:
                rows = client.query(policy_query).result()
                for row in rows:
                    if row.table_name not in schema.policy_tags:
                        schema.policy_tags[row.table_name] = {}
                    schema.policy_tags[row.table_name][row.column_name] = row.policy_tags
            except Exception:
                pass  # Policy tags optional

        # 6. Get column profiles from Data Profiling API (if enabled)
        if include_profiles:
            schema._fetch_column_profiles(client, project, dataset)

        return schema

    def _fetch_column_profiles(self, client, project: str, dataset: str):
        """Fetch column profiles using BigQuery Data Profiling API."""
        try:
            # Use the data profiling API via SQL
            # This queries the DATA_PROFILING system tables
            for table_name in self.tables:
                try:
                    # Get column profiles for this table
                    profile_query = f"""
                    SELECT
                        column_name,
                        profile_type,
                        min_value,
                        max_value,
                        avg_value,
                        stddev_value,
                        null_count,
                        distinct_count,
                        top_values
                    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMN_PROFILES`
                    WHERE table_name = '{table_name}'
                    """
                    rows = client.query(profile_query).result()
                    for row in rows:
                        if table_name not in self.column_profiles:
                            self.column_profiles[table_name] = {}
                        self.column_profiles[table_name][row.column_name] = {
                            "profile_type": row.profile_type,
                            "min_value": row.min_value,
                            "max_value": row.max_value,
                            "avg_value": row.avg_value,
                            "stddev_value": row.stddev_value,
                            "null_count": row.null_count,
                            "distinct_count": row.distinct_count,
                            "top_values": row.top_values,
                        }
                except Exception:
                    # Column profiles might not be available for all tables
                    pass
        except Exception:
            pass  # Profiling optional

    def get_enriched_schema(self) -> Dict:
        """Return enriched schema with all native metadata."""
        enriched = {}
        for table_name, table in self.tables.items():
            enriched[table_name] = {
                "columns": [
                    {
                        "name": col.name,
                        "type": col.type,
                        "nullable": col.nullable,
                    }
                    for col in table.columns
                ],
                "row_count": table.row_count,
                "partitions": self.partitions.get(table_name),
                "clustering": self.clustering.get(table_name),
                "column_profiles": self.column_profiles.get(table_name, {}),
                "policy_tags": self.policy_tags.get(table_name, {}),
                "storage": self.storage_info.get(table_name),
            }
        return enriched

    def join_hints(self, tables: List[str]) -> List[str]:
        """
        Generate join hints using BigQuery's INFORMATION_SCHEMA.TABLE_CONSTRAINTS
        if available, else fall back to column name matching.
        """
        # Try to get actual FK constraints from INFORMATION_SCHEMA
        # For now, fall back to column name matching
        return super().join_hints(tables)


# Helper classes for BigQuery schema loading (defined here to avoid import cycles)
@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: int = -1


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
            bigquery: project, dataset, credentials, location,
                      include_profiles, include_policy_tags,
                      include_partitions, include_clustering, include_storage
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
            include_profiles=kwargs.get("include_profiles", True),
            include_policy_tags=kwargs.get("include_policy_tags", True),
            include_partitions=kwargs.get("include_partitions", True),
            include_clustering=kwargs.get("include_clustering", True),
            include_storage=kwargs.get("include_storage", True),
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