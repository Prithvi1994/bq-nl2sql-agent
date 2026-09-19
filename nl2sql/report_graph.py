"""
LangGraph-based Report Generation Pipeline

Standard framework approach for report generation with:
- State management
- Human-in-the-loop
- Retry logic
- Observability
- Checkpointing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, Literal
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from nl2sql.agent import NL2SQLAgent
from nl2sql.schema_adapters import LocalFileSchema
from nl2sql.executor_duckdb import run_query_duckdb
from nl2sql.llm import create_llm
from nl2sql.prompts import report_generation_messages, report_review_messages, grade_report
from nl2sql.executor_duckdb import run_query_duckdb


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ReportStyle(str, Enum):
    EXECUTIVE = "executive_summary"
    DETAILED = "detailed_analysis"
    TECHNICAL = "technical_brief"
    SLIDES = "slide_deck"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class ReportState(TypedDict):
    # Input
    question: str
    style: ReportStyle
    model: str
    data_dir: str
    
    # SQL Generation Phase
    sql: Optional[str]
    sql_confidence: float
    sql_attempts: List[Dict]
    sql_error: Optional[str]
    
    # Execution Phase
    columns: List[str]
    rows: List[tuple]
    row_count: int
    exec_error: Optional[str]
    
    # Report Generation Phase
    report_sections: List[Dict]
    report_summary: str
    report_draft: str
    generation_attempts: int
    
    # Review Phase
    review_decision: Optional[ReviewDecision]
    review_feedback: Optional[str]
    review_count: int
    
    # Final Output
    final_report: Optional[Dict]
    
    # Metadata
    metadata: Dict[str, Any]


# ============================================================================
# NODES
# ============================================================================

def sql_generation_node(state: ReportState) -> ReportState:
    """Generate SQL from natural language question."""
    # Create agent from metadata (don't store agent in state)
    schema = LocalFileSchema.from_data_dir(state["data_dir"])
    agent = NL2SQLAgent(
        db_path=":memory:",
        llm=create_llm(state["model"]),
        schema=schema,
        executor=lambda db, sql, **kw: run_query_duckdb(sql, data_dir=state["data_dir"], **kw),
    )
    
    result = agent.ask(state["question"])
    
    state["sql"] = result.sql
    state["sql_confidence"] = result.confidence
    state["sql_attempts"] = [a.__dict__ for a in result.attempts]
    state["sql_error"] = result.attempts[-1].exec_error if result.status == "failed" else None
    
    if result.status == "failed":
        state["metadata"]["error_phase"] = "sql_generation"
    
    return state


def sql_execution_node(state: ReportState) -> ReportState:
    """Execute SQL and return structured data."""
    if not state["sql"]:
        state["exec_error"] = "No SQL generated"
        return state
    
    result = run_query_duckdb(
        state["sql"],
        data_dir=state["data_dir"],
        row_limit=1000,
        timeout_s=30.0
    )
    
    state["columns"] = result.columns
    state["rows"] = result.rows
    state["row_count"] = len(result.rows)
    state["exec_error"] = result.error
    
    if not result.ok:
        state["metadata"]["error_phase"] = "sql_execution"
    
    return state


def report_generation_node(state: ReportState) -> ReportState:
    """Generate report from structured data."""
    if state["exec_error"] or not state["rows"]:
        state["report_draft"] = f"Error: {state.get('exec_error') or state.get('sql_error')}"
        return state
    
    llm = create_llm(state["model"])
    
    # Format data for LLM
    data_preview = _format_data_for_llm(state["columns"], state["rows"], max_rows=100)
    
    # Handle style as string or enum
    style_value = state["style"].value if hasattr(state["style"], "value") else state["style"]
    
    messages = report_generation_messages(
        question=state["question"],
        sql=state["sql"],
        data_preview=data_preview,
        style=style_value,
        row_count=state["row_count"],
    )
    
    reply = llm.complete(messages)
    
    # Parse response
    sections = _parse_report_sections(reply)
    summary = _extract_summary(reply)
    
    state["report_sections"] = sections
    state["report_summary"] = summary
    state["report_draft"] = reply
    state["generation_attempts"] = state.get("generation_attempts", 0) + 1
    
    return state


def auto_grader_node(state: ReportState) -> ReportState:
    """
    Automated grader/eval loop that evaluates report quality and guides revision.
    
    Evaluates: groundedness, faithfulness, factual precision, completeness, style adherence.
    Returns: APPROVE (with score), REVISE (with specific feedback), or REJECT.
    """
    if not state.get("report_draft"):
        state["review_decision"] = ReviewDecision.REJECT
        state["review_feedback"] = "No report draft to evaluate"
        state["grader_scores"] = {}
        state["review_count"] = state.get("review_count", 0) + 1
        return state
    
    # Run grading evaluation
    grader_result = grade_report(
        question=state["question"],
        report=state["report_draft"],
        sql=state["sql"],
        columns=state["columns"],
        rows=state["rows"],
        style=state["style"].value if hasattr(state["style"], "value") else state["style"],
        model=state["model"],
    )
    
    # GraderResult is a dataclass, access attributes directly
    state["grader_scores"] = grader_result.scores
    state["grader_feedback"] = grader_result.feedback
    state["grader_decision"] = grader_result.decision
    state["review_count"] = state.get("review_count", 0) + 1
    
    # Map grader decision to review decision
    if grader_result.decision == "approve":
        state["review_decision"] = ReviewDecision.APPROVE
        state["review_feedback"] = f"Approved by grader (score: {grader_result.scores.get('overall', 0):.2f})"
    elif grader_result.decision == "revise":
        state["review_decision"] = ReviewDecision.REVISE
        state["review_feedback"] = grader_result.feedback
    else:
        state["review_decision"] = ReviewDecision.REJECT
        state["review_feedback"] = f"Rejected by grader: {grader_result.feedback}"
    
    return state


def revise_report_node(state: ReportState) -> ReportState:
    """Revise report based on human feedback."""
    if not state.get("review_feedback"):
        return state
    
    llm = create_llm(state["model"])
    
    # Create revision prompt
    messages = report_review_messages(
        question=state["question"],
        current_report=state["report_draft"],
        feedback=state["review_feedback"],
        style=state["style"].value,
    )
    
    reply = llm.complete(messages)
    sections = _parse_report_sections(reply)
    summary = _extract_summary(reply)
    
    state["report_sections"] = sections
    state["report_summary"] = summary
    state["report_draft"] = reply
    state["generation_attempts"] = state.get("generation_attempts", 0) + 1
    state["review_decision"] = None
    state["review_feedback"] = None
    
    return state


def finalize_report_node(state: ReportState) -> ReportState:
    """Finalize and structure the report."""
    # Handle style as string or enum
    style_value = state["style"].value if hasattr(state["style"], "value") else state["style"]
    
    state["final_report"] = {
        "title": f"Report: {state['question']}",
        "summary": state["report_summary"],
        "sections": state["report_sections"],
        "metadata": {
            "question": state["question"],
            "style": style_value,
            "model": state["model"],
            "sql": state["sql"],
            "row_count": state["row_count"],
            "sql_confidence": state["sql_confidence"],
            "review_count": state["review_count"],
            "generation_attempts": state["generation_attempts"],
        }
    }
    return state


def handle_error_node(state: ReportState) -> ReportState:
    """Handle errors gracefully."""
    error_phase = state["metadata"].get("error_phase", "unknown")
    error_msg = state.get("sql_error") or state.get("exec_error") or "Unknown error"
    
    state["final_report"] = {
        "title": f"Error Report: {state['question']}",
        "summary": f"Failed during {error_phase}: {error_msg}",
        "sections": [],
        "metadata": {
            "error": True,
            "error_phase": error_phase,
            "error_message": error_msg,
        }
    }
    return state


# ============================================================================
# ROUTING / CONDITIONAL EDGES
# ============================================================================

def should_continue_after_sql(state: ReportState) -> Literal["execute_sql", "handle_error"]:
    if state.get("sql_error"):
        return "handle_error"
    return "execute_sql"


def should_continue_after_execution(state: ReportState) -> Literal["generate_report", "handle_error"]:
    if state.get("exec_error") or not state.get("rows"):
        return "handle_error"
    return "generate_report"


def should_review(state: ReportState) -> Literal["auto_grader", "handle_error"]:
    if not state.get("report_draft"):
        return "handle_error"
    return "auto_grader"


def review_decision(state: ReportState) -> Literal["finalize", "revise", "handle_error"]:
    decision = state.get("review_decision")
    if decision == ReviewDecision.APPROVE:
        return "finalize"
    elif decision == ReviewDecision.REVISE:
        return "revise_report"
    elif decision == ReviewDecision.REJECT:
        return "handle_error"
    return "handle_error"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_report_graph(
    checkpointer: Optional[MemorySaver] = None,
    interrupt_before: Optional[List[str]] = None,
    enable_human_review: bool = True,
) -> StateGraph:
    """
    Build the report generation graph.
    
    Args:
        checkpointer: For checkpointing/persistence
        interrupt_before: Nodes to interrupt before (e.g., ["human_review"])
        enable_human_review: Whether to enable human review interrupt
    """
    workflow = StateGraph(ReportState)
    
    # Add nodes
    workflow.add_node("generate_sql", sql_generation_node)
    workflow.add_node("execute_sql", sql_execution_node)
    workflow.add_node("generate_report", report_generation_node)
    workflow.add_node("auto_grader", auto_grader_node)
    workflow.add_node("revise_report", revise_report_node)
    workflow.add_node("finalize_report", finalize_report_node)
    workflow.add_node("handle_error", handle_error_node)
    
    # Edges
    workflow.add_edge(START, "generate_sql")
    workflow.add_conditional_edges(
        "generate_sql",
        should_continue_after_sql,
        {"execute_sql": "execute_sql", "handle_error": "handle_error"}
    )
    workflow.add_conditional_edges(
        "execute_sql",
        should_continue_after_execution,
        {"generate_report": "generate_report", "handle_error": "handle_error"}
    )
    workflow.add_conditional_edges(
        "generate_report",
        should_review,
        {"auto_grader": "auto_grader", "handle_error": "handle_error"}
    )
    workflow.add_conditional_edges(
        "auto_grader",
        review_decision,
        {"finalize": "finalize_report", "revise": "revise_report", "handle_error": "handle_error"}
    )
    workflow.add_edge("revise_report", "auto_grader")
    workflow.add_edge("finalize_report", END)
    workflow.add_edge("handle_error", END)
    
    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # Only interrupt before human_review if enabled
    effective_interrupt = ["human_review"] if enable_human_review else None
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=effective_interrupt
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _format_data_for_llm(columns: List[str], rows: List[tuple], max_rows: int = 100) -> str:
    """Format data for LLM consumption."""
    if not rows:
        return "(no data)"
    
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in r) for r in rows[:max_rows])
    more = f"\n... ({len(rows)} rows total)" if len(rows) > max_rows else ""
    return f"{header}\n{body}{more}"


def _parse_report_sections(report_text: str) -> List[Dict]:
    """Parse report into sections."""
    import re
    sections = []
    
    # Try to find markdown headers
    parts = re.split(r'\n##\s+', report_text)
    for i, part in enumerate(parts):
        if i == 0 and not part.strip().startswith("#"):
            # First part might be intro
            if part.strip():
                sections.append({"title": "Introduction", "content": part.strip()})
        else:
            lines = part.strip().split("\n")
            title = lines[0] if lines else f"Section {i}"
            content = "\n".join(lines[1:]) if len(lines) > 1 else ""
            sections.append({"title": title, "content": content})
    
    return sections


def _extract_summary(report_text: str) -> str:
    """Extract or generate summary."""
    lines = report_text.strip().split("\n")
    for line in lines:
        if line.strip() and not line.startswith("#"):
            return line.strip()[:500]
    return "Report generated from data analysis."


# ============================================================================
# PUBLIC API
# ============================================================================

def run_report_generation(
    question: str,
    style: ReportStyle = ReportStyle.EXECUTIVE,
    model: str = "gpt-4o-mini",
    data_dir: str = "./data",
    thread_id: str = "default",
    enable_human_review: bool = True,
) -> Dict[str, Any]:
    """
    Run the report generation pipeline.
    
    Args:
        question: Natural language question
        style: Report style
        model: LLM model to use
        data_dir: Directory with CSV/Parquet/JSON files
        thread_id: Thread ID for checkpointing
        enable_human_review: Whether to pause for human review
    
    Returns:
        Final report dict
    """
    # Build graph
    checkpointer = MemorySaver()
    graph = build_report_graph(
        checkpointer=checkpointer,
        enable_human_review=enable_human_review,
    )
    
    # Initial state
    initial_state = ReportState(
        question=question,
        style=style,
        model=model,
        data_dir=data_dir,
        sql=None,
        sql_confidence=0.0,
        sql_attempts=[],
        sql_error=None,
        columns=[],
        rows=[],
        row_count=0,
        exec_error=None,
        report_sections=[],
        report_summary="",
        report_draft="",
        generation_attempts=0,
        review_decision=None,
        review_feedback=None,
        review_count=0,
        final_report=None,
        metadata={},
    )
    
    # Run
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config)
    
    return result.get("final_report", {})


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate report using LangGraph")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--style", choices=[s.value for s in ReportStyle], default="executive_summary")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--thread-id", default="report-1")
    parser.add_argument("--no-review", action="store_true")
    
    args = parser.parse_args()
    
    report = run_report_generation(
        question=args.question,
        style=ReportStyle(args.style),
        model=args.model,
        data_dir=args.data_dir,
        thread_id=args.thread_id,
        enable_human_review=not args.no_review,
    )
    
    import json
    print(json.dumps(report, indent=2, default=str))