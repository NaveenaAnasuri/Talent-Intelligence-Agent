from __future__ import annotations

from typing import TypedDict, Dict, Any, List, Optional
import pandas as pd
import networkx as nx


class WorkforceState(TypedDict, total=False):
    # inputs
    uploaded_files: Dict[str, bytes]
    question: str
    neo4j_status: str
    # tables
    employees: pd.DataFrame
    projects: pd.DataFrame
    performance: pd.DataFrame

    # graph
    graph: nx.MultiDiGraph

    # analytics outputs
    skill_gaps: List[Dict[str, Any]]
    concentration_risks: List[Dict[str, Any]]
    department_needs: List[Dict[str, Any]]
    risk_score: Dict[str, Any]
    simulation: Dict[str, Any]
    recommendations: Dict[str, Any]

    # report
    report_markdown: str
    report_pdf_path: Optional[str]

    # Q&A
    rag_context: str
    answer: str

    # scenario
    simulate_role: Optional[str]
    simulate_count: Optional[int]