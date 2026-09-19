# BQ-NL2SQL Agent Package
# 
# Core exports for the NL2SQL agent with grounded evaluation.
# Supports local testing via DuckDB and production BigQuery.

from .agent import NL2SQLAgent, AskResult, Attempt
from .guard import SQLGuardError, validate
from .llm import LLM, create_llm
from .prompts import generation_messages, repair_messages, judge_messages
from .schema import Schema, Table, Column, ForeignKey
from .schema_adapters import LocalFileSchema, BigQuerySchema, create_schema
from .executor import QueryResult, run_query
from .executor_duckdb import run_query_duckdb
from .evaluate import GoldenCase, CaseOutcome, EvalReport, evaluate, save_report
from .examples import ExampleBank

__all__ = [
    # Agent
    "NL2SQLAgent",
    "AskResult", 
    "Attempt",
    # Guard
    "SQLGuardError",
    "validate",
    # LLM
    "LLM",
    "create_llm",
    # Prompts
    "generation_messages",
    "repair_messages",
    "judge_messages",
    # Schema
    "Schema",
    "Table",
    "Column",
    "ForeignKey",
    # Schema adapters
    "LocalFileSchema",
    "BigQuerySchema",
    "create_schema",
    # Executor
    "QueryResult",
    "run_query",
    "run_query_duckdb",
    # Evaluation
    "GoldenCase",
    "CaseOutcome",
    "EvalReport",
    "evaluate",
    "save_report",
    # Examples
    "ExampleBank",
]

__version__ = "0.1.0"